import os
import random

import cv2
import numpy as np
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from path import Path
from skimage.io import imread

from torch.utils.data import Dataset
from tqdm import tqdm
from transforms3d.quaternions import mat2quat

from dataset.database import parse_database_name, get_object_center, Co3DResizeDatabase, get_database_split, \
    get_object_vert, get_diameter, NormalizedDatabase, normalize_pose, get_ref_point_cloud
from dataset.train_meta_info import name2database_names
from utils.base_utils import read_pickle, save_pickle, color_map_forward, project_points, transformation_compose_2d, \
    transformation_offset_2d, transformation_scale_2d, transformation_rotation_2d, transformation_apply_2d, \
    color_map_backward, sample_fps_points, transformation_crop, pose_inverse, pose_compose, transform_points_pose, \
    pose_apply
from utils.database_utils import select_reference_img_ids_fps, normalize_reference_views, \
    compute_normalized_view_correlation, look_at_crop, select_reference_img_ids_refinement
from utils.dataset_utils import set_seed
from utils.imgs_info import build_imgs_info, imgs_info_to_torch
from utils.pose_utils import scale_rotation_difference_from_cameras, let_me_look_at, let_me_look_at_2d, \
    estimate_pose_from_similarity_transform_compose


class MotionBlur(nn.Module):
    def __init__(self,max_ksize=10):
        super().__init__()
        self.max_ksize=max_ksize

    def forward(self, image):
        """
        @param image: [b,3,h,w] torch.float32 [0,1]
        @return:
        """
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, (self.max_ksize+1)/2)*2 + 1  # make sure is odd
        center = int((ksize-1)/2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        kernel = torch.from_numpy(kernel.astype(np.float32)) # [k,k]
        kernel = kernel.unsqueeze(0).unsqueeze(1) # [1,1,k,k]
        zeros = torch.zeros([1,1,ksize,ksize],dtype=torch.float32)
        kernel = torch.cat([torch.cat([kernel,zeros,zeros],0),torch.cat([zeros,kernel,zeros],0),torch.cat([zeros,zeros,kernel],0)],1)
        # kernel = kernel.repeat(1,1,1,1) # [3,3,k,k]
        # import ipdb; ipdb.set_trace()
        image = F.conv2d(image,kernel,padding=ksize//2)
        image = torch.clip(image,min=0.0,max=1.0)
        return image

class AdditiveShade(nn.Module):
    def __init__(self, nb_ellipses=5, transparency_range=(0.3, 0.5), kernel_size_range=(20, 50)):
        super().__init__()
        self.nb_ellipses=nb_ellipses
        self.transparency_range=transparency_range
        self.kernel_size_range=kernel_size_range

    def forward(self,image):
        h, w = image.shape[2:]
        min_dim = min(h, w) / 4
        mask = np.zeros([h, w], np.uint8)
        for i in range(self.nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, w - max_rad)  # center
            y = np.random.randint(max_rad, h - max_rad)
            angle = np.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*self.transparency_range)
        kernel_size = np.random.randint(*self.kernel_size_range)
        if np.random.random()<0.5: transparency = -transparency
        if (kernel_size % 2) == 0:  kernel_size += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        mask = torch.from_numpy((1 - transparency * mask / 255.).astype(np.float32))
        image = image * mask[None,None,:,:]
        image = torch.clip(image,min=0.0,max=1.0)
        return image

COCO_IMAGE_ROOT = 'data/coco/train2017'
def get_coco_image_fn_list():
    if Path('data/COCO_list.pkl').exists():
        return read_pickle('data/COCO_list.pkl')
    img_list = os.listdir(COCO_IMAGE_ROOT)
    img_list = [img for img in img_list if img.endswith('.jpg')]
    save_pickle(img_list, 'data/COCO_list.pkl')
    return img_list

def get_background_image_coco(fn, h, w):
    back_img = imread(f'{COCO_IMAGE_ROOT}/{fn}')
    h1, w1 = back_img.shape[:2]
    if h1 > h and w1 > w:
        hb = np.random.randint(0, h1 - h)
        wb = np.random.randint(0, w1 - w)
        back_img = back_img[hb:hb + h, wb:wb + w]
    else:
        back_img = cv2.resize(back_img,(w,h),interpolation=cv2.INTER_LINEAR)
    if len(back_img.shape)==2:
        back_img = np.repeat(back_img[:,:,None],3,2)
    return back_img[:,:,:3]

def crop_bbox_resize_to_target_size(img, mask, bbox, target_size, margin_ratio,
                                    scale_aug=None, rotation_aug=None, offset_aug=None):
    """
    scale -> rotation -> offset
    @param img:
    @param mask:
    @param bbox:
    @param target_size:
    @param margin_ratio:
    @param scale_aug:
    @param rotation_aug:
    @param offset_aug:
    @return:
    """
    center = bbox[:2] + bbox[2:] / 2
    bw, bh = bbox[2:]
    rotation_val = 0.0
    if bw==0 or bh==0:
        scale_val = 1.0
    else:
        scale_val = target_size / (max(bw, bh) * (1 + margin_ratio))

    M = transformation_offset_2d(-center[0], -center[1])
    M = transformation_compose_2d(M, transformation_scale_2d(scale_val))
    if scale_aug is not None:
        M = transformation_compose_2d(M, transformation_scale_2d(scale_aug))
        scale_val *= scale_aug
    if rotation_aug is not None:
        M = transformation_compose_2d(M, transformation_rotation_2d(rotation_aug))
        rotation_val += rotation_aug
    offset_val = np.asarray([0,0],np.float32)
    if offset_aug is not None:
        M = transformation_compose_2d(M, transformation_offset_2d(offset_aug[0], offset_aug[1]))
        offset_val += offset_aug
    offset1 = transformation_offset_2d(target_size / 2, target_size / 2)
    M = transformation_compose_2d(M, offset1)

    img = cv2.warpAffine(img, M, (target_size, target_size), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask.astype(np.float32), M, (target_size, target_size), flags=cv2.INTER_LINEAR)
    return img, mask, M

def _check_detection(que_imgs_info, ref_imgs_info, gt_ref_idx, scale_diff, rotation_diff, name):
    from utils.draw_utils import concat_images_list
    from skimage.io import imsave
    qn = que_imgs_info['imgs'].shape[0]
    que_imgs = color_map_backward(que_imgs_info['imgs'].numpy()).transpose([0,2,3,1])
    ref_imgs = color_map_backward(ref_imgs_info['imgs'].numpy()).transpose([0,2,3,1])
    _, hr, wr, _ = ref_imgs.shape
    imgs, whole_imgs = [], []
    for qi in range(qn):
        # ref to que
        scale = 1/scale_diff.numpy()[qi]
        rotation = -rotation_diff.numpy()[qi]
        center = que_imgs_info['cens'][qi]
        M = transformation_offset_2d(-center[0], -center[1])
        M = transformation_compose_2d(M, transformation_rotation_2d(rotation))
        M = transformation_compose_2d(M, transformation_scale_2d(scale))
        M = transformation_compose_2d(M, transformation_offset_2d(wr/2,hr/2))
        warp_img = cv2.warpAffine(que_imgs[qi],M,(wr,hr))

        M = transformation_offset_2d(-center[0], -center[1])
        M = transformation_compose_2d(M, transformation_offset_2d(wr/2,hr/2))
        warp_img2 = cv2.warpAffine(que_imgs[qi],M,(wr,hr))

        ref_img = ref_imgs[gt_ref_idx[qi]]
        imgs.append(concat_images_list(ref_img, warp_img, warp_img2, vert=True))
        whole_imgs.append(que_imgs[qi])

    imsave(name,concat_images_list(concat_images_list(*imgs),concat_images_list(*ref_imgs),concat_images_list(*whole_imgs),vert=True))
    print('check mode is on!!')

class Gen6DTrainDataset(Dataset):
    default_cfg={
        'batch_size': 8,
        "use_database_sample_prob": False,
        "database_sample_prob": [100, 10, 30, 10, 10],
        'database_names': ['co3d_train', 'gso_train_128', 'shapenet_train', 'linemod_train', 'genmop_train'],

        "resolution": 128,
        "reference_num": 32,
        "co3d_margin_ratio": 0.3,
    }
    def __init__(self, cfg, is_train):
        self.cfg = {**self.default_cfg, **cfg}
        self.is_train = is_train

        self.database_names = []
        self.database_set_names = []
        self.database_set_name2names = {}
        for name in self.cfg['database_names']:
            self.database_names += name2database_names[name]
            self.database_set_names.append(name)
            self.database_set_name2names[name] = name2database_names[name]

        self.name2database = {}
        for name in tqdm(self.database_names):
            self.name2database[name] = parse_database_name(name)
            if name.startswith('genmop'):
                test_name = name.replace('test','ref')
                self.name2database[test_name] = parse_database_name(test_name)

        self.cum_que_num = np.cumsum([len(self.name2database[name].get_img_ids()) for name in self.database_names])
        self.background_img_list = get_coco_image_fn_list() # get_SUN397_image_fn_list()
        self.photometric_augment_modules = [
            T.GaussianBlur(3),
            T.ColorJitter(brightness=0.3),
            T.ColorJitter(contrast=0.2),
            T.ColorJitter(hue=0.05),
            T.ColorJitter(saturation=0.3),
            MotionBlur(5),
            AdditiveShade(),
        ]

    def __len__(self):
        if self.is_train:
            return 999999
        else:
            return self.cum_que_num[-1]

    def _select_query(self,index):
        if self.is_train:
            if self.cfg['use_database_sample_prob']:
                probs = np.asarray(self.cfg['database_sample_prob'])
                probs = probs/np.sum(probs)
                database_set_name = np.random.choice(self.database_set_names,p=probs)
                names = self.database_set_name2names[database_set_name]
                database = self.name2database[np.random.choice(names)]
            else:
                database = self.name2database[self.database_names[np.random.randint(0,len(self.database_names))]]
            img_ids = database.get_img_ids()
            random.shuffle(img_ids)
            que_ids = img_ids[:self.cfg['batch_size']]
        else:
            data_id = np.searchsorted(self.cum_que_num, index, 'right')
            database = self.name2database[self.database_names[data_id]]
            image_id_back = self.cum_que_num[data_id] - index
            que_ids = [database.get_img_ids()[-image_id_back]] # only use single image in testing
        return database, que_ids

    def _add_background(self, imgs, masks, same_background_prob):
        """
        @param imgs:   [b,3,h,w] in [0,1] torch.tensor
        @param masks:
        @param same_background_prob:
        @return:
        """
        # imgs = imgs_info['imgs']
        # masks = imgs_info['masks']
        qn, _, h, w = imgs.shape
        back_imgs = []
        if np.random.random() < same_background_prob:
            fn = self.background_img_list[np.random.randint(0, len(self.background_img_list))]
            back_img_global = get_background_image_coco(fn, h, w)
        else:
            back_img_global = None

        for qi in range(qn):
            if back_img_global is None:
                fn = self.background_img_list[np.random.randint(0, len(self.background_img_list))]
                back_img = get_background_image_coco(fn, h, w)
            else:
                back_img = back_img_global
            back_img = color_map_forward(back_img)
            if len(back_img.shape)==2:
                back_img = np.repeat(back_img[:,:,None],3,2)
            back_img = torch.from_numpy(back_img).permute(2,0,1)
            back_imgs.append(back_img)
        back_imgs = torch.stack(back_imgs,0)
        masks = masks.float()
        imgs = imgs*masks + (1 - masks)*back_imgs
        return imgs

    def _build_ref_imgs_info(self, database, ref_ids):
        if database.database_name.startswith('gso') or database.database_name.startswith('shapenet'):
            ref_imgs_info = build_imgs_info(database, ref_ids)
            rfn = len(ref_ids)
            M = np.concatenate([np.eye(2),np.zeros([2,1])],1)
            ref_imgs_info['Ms'] = np.repeat(M[None,:],rfn,0)
            ref_imgs_info['ref_ids'] = np.asarray(ref_ids)
            object_center = get_object_center(database)
            # add object center to imgs_info
            ref_imgs_info['cens'] = [project_points(object_center[None],pose,K)[0][0] for pose,K in zip(ref_imgs_info['poses'],ref_imgs_info['Ks'])] # object center
            ref_imgs_info['cens'] = np.asarray(ref_imgs_info['cens'])
        elif database.database_name.startswith('co3d'):
            t = self.cfg['resolution']
            m = self.cfg['co3d_margin_ratio']
            imgs, masks, Ms = [], [], []
            for ref_id in ref_ids:
                assert(isinstance(database, Co3DResizeDatabase))
                img = database.get_image(ref_id)
                mask = database.get_mask(ref_id)
                bbox = database.get_bbox(ref_id)
                img, mask, M = crop_bbox_resize_to_target_size(img, mask, bbox, t, m)

                imgs.append(img)
                masks.append(mask)
                Ms.append(M)

            imgs = np.stack(imgs, 0)
            imgs = color_map_forward(imgs)
            masks = np.stack(masks, 0)
            Ms = np.stack(Ms,0)
            poses = np.asarray([database.get_pose(img_id) for img_id in ref_ids])
            Ks = np.asarray([database.get_K(img_id) for img_id in ref_ids])
            cens = np.repeat(np.asarray([t/2,t/2])[None,:],len(ref_ids),0)  # add object center to imgs_info
            ref_imgs_info = {'ref_ids': np.asarray(ref_ids), 'imgs': imgs.transpose([0,3,1,2]), 'masks': masks[:,None,:,:], 'Ms': Ms, 'poses': poses, 'Ks': Ks, 'cens': cens}
        elif database.database_name.startswith('linemod') or database.database_name.startswith('genmop'):
            ref_ids_all = database.get_img_ids()
            res = self.cfg['resolution']
            ref_num = self.cfg['reference_num']
            ref_ids = select_reference_img_ids_fps(database, ref_ids_all, ref_num, random_fps=self.is_train)
            imgs, masks, Ks, poses, _ = normalize_reference_views(database, ref_ids, res, 0.05)

            M = np.concatenate([np.eye(2), np.zeros([2, 1])], 1)
            rfn, h, w, _ = imgs.shape
            object_center = get_object_center(database)
            cens = np.asarray([project_points(object_center[None], pose, K)[0][0] for pose, K in zip(poses, Ks)],np.float32)
            ref_imgs_info = {
                'imgs': color_map_forward(imgs).transpose([0,3,1,2]),
                'masks': np.ones([rfn,1,h,w],dtype=np.float32),
                'ref_ids': [None for _ in range(rfn)],
                'Ms': np.repeat(M[None,:], rfn, 0),
                'poses': poses.astype(np.float32),
                'Ks': Ks.astype(np.float32),
                'cens': cens, # rfn, 2
            }
        else:
            raise NotImplementedError
        return ref_imgs_info

    def _photometric_augment(self, imgs_info, aug_prob):
        if len(imgs_info['imgs'].shape)==3:
            if np.random.random() < aug_prob:
                ids = np.random.choice(np.arange(len(self.photometric_augment_modules)), np.random.randint(1, 4), False)
                for idx in ids: imgs_info['imgs'] = self.photometric_augment_modules[idx](imgs_info['imgs'][None])[0]
        else:
            qn = imgs_info['imgs'].shape[0]
            for qi in range(qn):
                if np.random.random()<aug_prob:
                    ids = np.random.choice(np.arange(len(self.photometric_augment_modules)), np.random.randint(1, 4), False)
                    for idx in ids: imgs_info['imgs'][qi:qi + 1] = self.photometric_augment_modules[idx](imgs_info['imgs'][qi:qi + 1])

    def _photometric_augment_imgs(self, imgs, aug_prob):
        qn = imgs.shape[0]
        for qi in range(qn):
            if np.random.random()<aug_prob:
                ids = np.random.choice(np.arange(len(self.photometric_augment_modules)), np.random.randint(1, 4), False)
                for idx in ids: imgs[qi:qi+1] = self.photometric_augment_modules[idx](imgs[qi:qi + 1])
        return imgs

    def __getitem__(self, index):
        raise NotImplementedError

def add_object_to_background(img, mask, back_img, max_obj_ratio=0.5):
    """

    @param img: [0,1] in float32
    @param mask:
    @param back_img:
    @param max_obj_ratio:
    @return:
    """
    img_out = np.copy(back_img)
    h1, w1 = img_out.shape[:2]

    # get compact region
    ys, xs = np.nonzero(mask.astype(np.bool))
    min_x, max_x, min_y, max_y = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
    img = img[min_y:max_y, min_x:max_x]
    mask = mask[min_y:max_y, min_x:max_x]
    h, w = img.shape[:2]

    # if too large, we resize it
    if max(h, w)/max(h1, w1)>max_obj_ratio:
        ratio = max(h1, w1) * np.random.uniform(0.1, max_obj_ratio) / max(h, w)
        h, w = int(round(ratio*h)),int(round(ratio*w))
        mask = cv2.resize(mask.astype(np.uint8),(w, h), interpolation=cv2.INTER_LINEAR)>0
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    h0, w0 = np.random.randint(0, h1 - h), np.random.randint(0, w1 - w)
    raw = img_out[h0:h + h0, w0:w + w0]
    img_out[h0:h + h0, w0:w + w0] = img * mask.astype(np.float32)[:, :, None] + \
                                    raw * (1 - mask[:, :, None].astype(np.float32))

    mask_out = np.zeros([h1, w1], dtype=np.bool)
    mask_out[h0:h + h0, w0:w + w0] = mask.astype(np.bool)
    bbox_out = np.asarray([w0, h0, w, h], np.float32)
    return img_out, mask_out, bbox_out

def transformation_decompose_scale(M):
    return np.sqrt(np.linalg.det(M[:2,:2]))

def transformation_decompose_rotation(M):
    return np.arctan2(M[1,0],M[0,0])

def get_ref_ids(database, ref_view_type):
    if database.database_name.startswith('linemod') or database.database_name.startswith('genmop'):
        return []

    # sample farthest points for these datasets
    if ref_view_type.startswith('fps'):
        anchor_num = int(ref_view_type.split('_')[-1])
        img_ids = database.get_img_ids()
        poses = [database.get_pose(img_id) for img_id in img_ids]
        cam_pts = np.asarray([(pose[:,:3].T @ pose[:,3:])[...,0] for pose in poses],np.float32)
        indices = sample_fps_points(cam_pts, anchor_num, False, True)
        ref_ids = np.asarray(img_ids)[indices]
    else:
        raise NotImplementedError
    return ref_ids

class DetectionTrainDataset(Gen6DTrainDataset):
    det_default_cfg={


        'ref_type': 'fps_32',
        "detector_scale_range": [-0.5,1.2],
        "detector_rotation_range": [-22.5,22.5],

        # only used in validation
        'background_database_name': 'gso_train_128',
        'background_database_num': 32,

        # query image resolution
        "query_resolution": 512,

        # co3d settings of generate query iamge
        'que_add_background_objects': True,
        'que_background_objects_num': 2,
        'que_background_objects_ratio': 0.3,

        'offset_type': 'random',
        # "detector_offset_range": [-0.1,0.1],
        "detector_offset_std": 3,
        'detector_real_aug_rot': True,
    }
    def __init__(self, cfg, is_train):
        cfg={**self.det_default_cfg, **cfg}
        super(DetectionTrainDataset, self).__init__(cfg, is_train)
        if is_train:
            self.name2back_database = {k:v for k,v in self.name2database.items() if (not k.startswith('genmop')) and (not k.startswith('linemod'))}
        else:
            random.seed(1234)
            background_names = name2database_names[self.cfg['background_database_name']]
            random.shuffle(background_names)
            background_names = background_names[:self.cfg['background_database_num']]
            self.name2back_database = {name:parse_database_name(name) for name in background_names}
            self.name2back_database.update(self.name2database)
        self.back_names = [name for name in self.name2back_database.keys()]

    def get_offset(self, output_resolution, M, mask):
        if self.cfg['offset_type'] == 'random':
            # add random transformations and put objects on random regions
            ys, xs = np.nonzero(mask)
            min_x, max_x, min_y, max_y = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
            corners = np.asarray([[min_x, min_y,],[min_x, max_y,],[max_x, max_y,],[max_x, min_y,]],np.float32)
            corners_ = transformation_apply_2d(M, corners)
            min_x_, min_y_ = np.min(corners_, 0)
            max_x_, max_y_ = np.max(corners_, 0)
            if (max_x_ - min_x_ >= output_resolution) or (max_y_ - min_y_ >= output_resolution):
                offset_x = output_resolution / 2; offset_y = output_resolution / 2
                # actually impossible here, because we already build ref imgs info to resize them to 128
                raise NotImplementedError
            else:
                offset_x = np.random.uniform(-min_x_, output_resolution - max_x_)
                offset_y = np.random.uniform(-min_y_, output_resolution - max_y_)
            M = transformation_compose_2d(M, transformation_offset_2d(offset_x, offset_y))
        elif self.cfg['offset_type'] == 'center':
            # approximately keep the object in the center
            # off_min, off_max = self.cfg['detector_offset_range']
            # off_x, off_y = output_resolution * np.random.uniform(off_min,off_max,2)
            off_x, off_y = np.random.normal(0,self.cfg['detector_offset_std'],2)
            M = transformation_compose_2d(M, transformation_offset_2d(off_x, off_y))
            M = transformation_compose_2d(M, transformation_offset_2d(output_resolution/2, output_resolution/2))
        else:
            raise NotImplementedError
        return M

    def _build_que_imgs_info(self, database, que_ids):
        if database.database_name.startswith('linemod') or database.database_name.startswith('genmop'):
            # todo: add random background or rotation for linemod dataset
            que_imgs_info = build_imgs_info(database, que_ids, has_mask=False)
            poses, Ks, imgs = que_imgs_info['poses'], que_imgs_info['Ks'], que_imgs_info['imgs']
            qn, _, h, w = imgs.shape
            M = np.concatenate([np.eye(2), np.zeros([2, 1])], 1)
            Ms = np.repeat(M[None], qn, 0)
            object_center = get_object_center(database)
            cens = np.asarray([project_points(object_center[None], pose, K)[0][0] for pose, K in zip(poses, Ks)], np.float32)
            que_imgs_info.update({
                'Ms': Ms,
                'cens': np.stack(cens, 0),        # [qn,2]
                'que_ids': np.asarray(que_ids),
            })
        else:
            que_imgs_info = self._build_ref_imgs_info(database, que_ids)
            imgs, masks, Ms, cens = [], [], [], []
            q = self.cfg['query_resolution']
            for qi in range(len(que_ids)):
                img = color_map_backward(que_imgs_info['imgs'][qi].transpose([1,2,0]))
                mask = que_imgs_info['masks'][qi][0]
                center = que_imgs_info['cens'][qi]

                scale_aug = 2 ** np.random.uniform(*self.cfg['detector_scale_range'])
                rotation_aug = np.random.uniform(*self.cfg['detector_rotation_range'])
                rotation_aug = np.deg2rad(rotation_aug)

                M = transformation_offset_2d(-center[0],-center[1]) # offset to object center
                M = transformation_compose_2d(M, transformation_scale_2d(scale_aug))
                M = transformation_compose_2d(M, transformation_rotation_2d(rotation_aug))
                M = self.get_offset(q,M,mask) # get offset
                if database.database_name.startswith('co3d'):
                    # warp co3d image from original image
                    M_init = que_imgs_info['Ms'][qi]
                    img_init = database.get_image(que_ids[qi])
                    mask_init = database.get_mask(que_ids[qi])
                    M_ = transformation_compose_2d(M_init, M)
                    img = cv2.warpAffine(img_init, M_, (q, q), flags=cv2.INTER_LINEAR)
                    mask = cv2.warpAffine(mask_init.astype(np.uint8), M_, (q, q), flags=cv2.INTER_LINEAR)
                else:
                    img = cv2.warpAffine(img, M, (q, q), flags=cv2.INTER_LINEAR)
                    mask = cv2.warpAffine(mask.astype(np.uint8), M, (q, q), flags=cv2.INTER_LINEAR)

                # add random background if it is a synthetic data or 80% probability with co3d dataset
                if database.database_name.startswith('gso') or database.database_name.startswith('shapenet') or np.random.random() < 0.8:
                    fn = self.background_img_list[np.random.randint(0, len(self.background_img_list))]
                    back_img = get_background_image_coco(fn, q, q).astype(np.float32)
                    mask_ = mask[:,:,None]
                    img = back_img * (1 - mask_) + img * mask_

                # add random objects
                img = color_map_forward(img)
                mask = mask.astype(np.bool)
                # add background objects
                if self.cfg['que_add_background_objects']:
                    img_ = self._add_background_objects(img, database, self.cfg['que_background_objects_num'], self.cfg['que_background_objects_ratio'])
                    # prevent background object overwrite foreground object
                    mask_ = mask.astype(np.float32)[:, :, None]
                    img = img * mask_ + img_ * (1 - mask_)

                imgs.append(img)
                masks.append(mask)
                cens.append(transformation_apply_2d(M, np.asarray([center]))[0]) # [2]

                # apply initial transformations
                M_in = que_imgs_info['Ms'][qi]
                Ms.append(transformation_compose_2d(M_in, M))


            que_imgs_info = {
                'imgs': np.stack(imgs, 0).transpose([0, 3, 1, 2]),
                'masks': np.stack(masks, 0)[:,None],
                'Ms': np.stack(Ms, 0),     # [qn,2,3]
                'cens': np.stack(cens, 0), # [qn,2]
                'que_ids': np.asarray(que_ids),
                'poses': que_imgs_info['poses'],
                'Ks': que_imgs_info['Ks'],
            }
        return que_imgs_info

    def _add_background_objects(self, que_img, database, object_num, max_background_object_size=0.5):
        """

        @param que_img: [0,1] in float32
        @param database:
        @return:
        """
        if object_num > 0:
            for obj_id in range(object_num):
                while True:
                    random_database = self.name2back_database[self.back_names[np.random.randint(0, len(self.back_names))]]
                    if random_database.database_name != database.database_name: break
                img_id = np.random.choice(random_database.get_img_ids())
                img = random_database.get_image(img_id)
                img = color_map_forward(img)
                mask = random_database.get_mask(img_id)
                que_img, _, _ = add_object_to_background(img, mask, que_img, max_background_object_size)
        return que_img

    @staticmethod
    def que_ref_scale_rotation_from_poses(center, ref_imgs_info, que_imgs_info):
        """
        call this function before to torch
        @param center: used in
        @param ref_imgs_info:
        @param que_imgs_info:
        @return:
            1. transformation from reference to query
               rotate(scale(ref)) = que
            2. gt_ref_ids [qn,] the ground truth reference view index for every query view
        """

        ref_poses = ref_imgs_info['poses']
        que_poses = que_imgs_info['poses']
        ref_Ks = ref_imgs_info['Ks']
        que_Ks = que_imgs_info['Ks']

        # select nearest views
        corr = compute_normalized_view_correlation(que_poses, ref_poses, center, False)
        gt_ref_ids = np.argmax(corr, 1)  # qn

        scale_diff, rotation_diff = scale_rotation_difference_from_cameras(
            ref_poses[gt_ref_ids], que_poses, ref_Ks[gt_ref_ids], que_Ks, center) # from ref to que

        ref_scales = np.asarray([transformation_decompose_scale(M) for M in ref_imgs_info['Ms'][gt_ref_ids]])
        que_scales = np.asarray([transformation_decompose_scale(M) for M in que_imgs_info['Ms']])
        ref_rotations = np.asarray([transformation_decompose_rotation(M) for M in ref_imgs_info['Ms'][gt_ref_ids]])
        que_rotations = np.asarray([transformation_decompose_rotation(M) for M in que_imgs_info['Ms']])
        scale = scale_diff * que_scales / ref_scales
        rotation = -ref_rotations + rotation_diff + que_rotations
        return scale, rotation, gt_ref_ids, corr

    @staticmethod
    def que_ref_scale_rotation_from_index(database, ref_imgs_info, que_imgs_info):
        ref_ids = np.asarray(ref_imgs_info['ref_ids'])  # rfn
        que_ids = np.asarray(que_imgs_info['que_ids'])  # qn
        diff = np.abs(que_ids[:, None].astype(np.int32) - ref_ids[None, :].astype(np.int32))
        gt_ref_ids = np.argmin(diff, 1)

        ref_scales = np.asarray([transformation_decompose_scale(M) for M in ref_imgs_info['Ms'][gt_ref_ids]])
        que_scales = np.asarray([transformation_decompose_scale(M) for M in que_imgs_info['Ms']])
        ref_rotations = np.asarray([transformation_decompose_rotation(M) for M in ref_imgs_info['Ms'][gt_ref_ids]])
        que_rotations = np.asarray([transformation_decompose_rotation(M) for M in que_imgs_info['Ms']])

        for qi in range(len(gt_ref_ids)):
            ref_id = ref_ids[gt_ref_ids[qi]]
            que_id = que_ids[qi]
            # record scale for the computation of scale difference
            if isinstance(database, Co3DResizeDatabase):
                # since we already resize co3d dataset when resizing
                ref_scales[qi] *= database.ratios[ref_id]
                que_scales[qi] *= database.ratios[que_id]

        # ref to query
        scale = que_scales / ref_scales
        rotation = que_rotations - ref_rotations
        return scale, rotation, gt_ref_ids

    def _compute_scale_rotation_target(self, database, ref_imgs_info, que_imgs_info):
        if database.database_name.startswith('gso') or \
            database.database_name.startswith('shapenet') or \
            database.database_name.startswith('linemod') or \
            database.database_name.startswith('genmop'):
            center = get_object_center(database)
            scale_diff, rotation_diff, gt_ref_ids, _ = \
                self.que_ref_scale_rotation_from_poses(center, ref_imgs_info, que_imgs_info)
        elif database.database_name.startswith('co3d'):
            scale_diff, rotation_diff, gt_ref_ids = \
                self.que_ref_scale_rotation_from_index(database, ref_imgs_info, que_imgs_info)
        else:
            raise NotImplementedError
        return scale_diff, rotation_diff, gt_ref_ids

    def add_background_or_not(self, que_database):
        add_background=False
        if que_database.database_name.startswith('shapenet') or que_database.database_name.startswith('gso'):
            add_background=True
        elif que_database.database_name.startswith('co3d'):
            if np.random.random()<0.75:
                add_background = True
        elif que_database.database_name.startswith('linemod'):
            add_background = False
        elif que_database.database_name.startswith('genmop'):
            add_background = False
        else:
            raise NotImplementedError
        return add_background

    def __getitem__(self, index):
        set_seed(index,self.is_train)
        que_database, que_ids = self._select_query(index)
        if que_database.database_name.startswith('genmop'):
            que_name = que_database.database_name
            ref_database = self.name2database[que_name.replace('test', 'ref')]
        else:
            ref_database = que_database
        ref_ids = get_ref_ids(ref_database, self.cfg['ref_type'])

        ref_imgs_info = self._build_ref_imgs_info(ref_database, ref_ids)
        que_imgs_info = self._build_que_imgs_info(que_database, que_ids)

        # compute scale and rotation difference
        scale_diff, rotation_diff, gt_ref_idx = \
            self._compute_scale_rotation_target(que_database,ref_imgs_info,que_imgs_info)

        ref_imgs_info.pop('ref_ids'); que_imgs_info.pop('que_ids')
        ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        que_imgs_info = imgs_info_to_torch(que_imgs_info)


        if self.is_train and self.add_background_or_not(que_database):
            ref_imgs_info['imgs'] = self._add_background(ref_imgs_info['imgs'], ref_imgs_info['masks'], 0.5)

        if self.is_train:
            self._photometric_augment(que_imgs_info, 0.8)
            self._photometric_augment(ref_imgs_info, 0.8)

        scale_diff = torch.from_numpy(scale_diff.astype(np.float32))
        rotation_diff = torch.from_numpy(rotation_diff.astype(np.float32))
        gt_ref_idx = torch.from_numpy(gt_ref_idx.astype(np.int32))

        # _check_detection(que_imgs_info, ref_imgs_info, gt_ref_idx, scale_diff, rotation_diff, f'data/vis_val/{index}.jpg')

        return {'que_imgs_info': que_imgs_info, 'ref_imgs_info': ref_imgs_info,
                'gt_ref_idx': gt_ref_idx, 'scale_diff': scale_diff, 'rotation_diff': rotation_diff}

class DetectionValDataset(Dataset):
    default_cfg={
        "test_database_name": 'linemod/cat',
        "ref_database_name": 'linemod/cat',
        "test_split_type": "linemod_val",
        "ref_split_type": "linemod_val",
        "detector_ref_num": 32,
        "detector_ref_res": 128,
    }
    def __init__(self, cfg, is_train):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__()
        assert(not is_train)
        self.test_database = parse_database_name(self.cfg['test_database_name'])
        self.ref_database = parse_database_name(self.cfg['ref_database_name'])
        ref_ids, _ = get_database_split(self.ref_database,self.cfg['ref_split_type'])
        _, self.test_ids = get_database_split(self.test_database,self.cfg['test_split_type'])

        ref_ids = select_reference_img_ids_fps(self.ref_database, ref_ids, self.cfg['detector_ref_num'])
        # ref_imgs_new, ref_masks_new, ref_Ks_new, ref_poses_new, ref_Hs
        ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs = \
            normalize_reference_views(self.ref_database, ref_ids, self.cfg['detector_ref_res'], 0.05)

        self.ref_info={
            'poses': torch.from_numpy(ref_poses.astype(np.float32)),
            'Ks': torch.from_numpy(ref_Ks.astype(np.float32)),
            'imgs': torch.from_numpy(color_map_forward(ref_imgs)).permute(0,3,1,2)
        }
        self.center = get_object_center(self.ref_database).astype(np.float32)
        self.res = self.cfg['detector_ref_res']

    def __getitem__(self, index):
        ref_imgs_info = self.ref_info.copy()
        img_id = self.test_ids[index]
        que_img = self.test_database.get_image(img_id)

        center_np = self.center
        que_poses = self.test_database.get_pose(img_id)[None,]
        que_Ks = self.test_database.get_K(img_id)[None,]
        que_cen = project_points(center_np[None], que_poses[0], que_Ks[0])[0][0]
        ref_poses = ref_imgs_info['poses'].numpy()
        ref_Ks = ref_imgs_info['Ks'].numpy()

        corr = compute_normalized_view_correlation(que_poses, ref_poses, center_np, False)
        gt_ref_ids = np.argmax(corr, 1)  # qn
        scale_diff, angle_diff = scale_rotation_difference_from_cameras(ref_poses[gt_ref_ids], que_poses, ref_Ks[gt_ref_ids], que_Ks, center_np)

        que_imgs_info = {
            'imgs': torch.from_numpy(color_map_forward(que_img)[None]).permute(0,3,1,2),
            'poses': torch.from_numpy(que_poses.astype(np.float32)),
            'Ks': torch.from_numpy(que_Ks.astype(np.float32)),
            'cens': torch.from_numpy(que_cen[None]),
        }
        scale_diff = torch.from_numpy(scale_diff.astype(np.float32))
        angle_diff = torch.from_numpy(angle_diff.astype(np.float32))
        gt_ref_ids = torch.from_numpy(gt_ref_ids.astype(np.int32))
        return {'que_imgs_info': que_imgs_info, 'ref_imgs_info': ref_imgs_info, 'gt_ref_idx': gt_ref_ids, 'scale_diff': scale_diff, 'rotation_diff': angle_diff}

    def __len__(self):
        return len(self.test_ids)

def _check_selection(que_imgs_info,ref_imgs,ref_vp_scores,angles_r2q,index):
    from utils.draw_utils import concat_images_list
    from skimage.io import imsave
    que_imgs = color_map_backward(que_imgs_info['imgs'].cpu().numpy()).transpose([0,2,3,1]) # qn,h,w,3
    ref_imgs = color_map_backward(ref_imgs.cpu().numpy()).transpose([0,1,3,4,2]) # an,rfn,h,w,3
    angles_r2q = angles_r2q.cpu().numpy()
    ref_vp_scores = ref_vp_scores.cpu().numpy() # qn,rfn
    ref_vp_idx = np.argsort(-ref_vp_scores,1) # qn,rfn
    qn, h, w, _ = que_imgs.shape
    for qi in range(1):
        M = transformation_offset_2d(-w/2,-h/2)
        M = transformation_compose_2d(M, transformation_rotation_2d(-angles_r2q[qi]))
        M = transformation_compose_2d(M, transformation_offset_2d(w/2, h/2))

        warp_img = cv2.warpAffine(que_imgs[qi], M, (w,h), cv2.INTER_LINEAR)
        ori_img = que_imgs[qi]
        cur_imgs = [concat_images_list(ori_img, warp_img)]
        for k in range(5):
            cur_imgs.append(concat_images_list(*[ref_imgs[ai, ref_vp_idx[qi,k]] for ai in range(5)]))
        imsave(f'data/vis_val/{index}-{qi}.jpg',concat_images_list(*cur_imgs,vert=True))

    print('check mode is on!')

class SelectionTrainDataset(Gen6DTrainDataset):
    default_cfg_v2 = {
        'ref_type': 'fps_32',
        'selector_scale_range': [-0.1, 0.1],
        'selector_angle_range': [-90, 90],
        'selector_angles': [-90, -45, 0, 45, 90],
        'selector_real_aug': False,
    }
    def __init__(self, cfg, is_train):
        cfg = {**self.default_cfg_v2,**cfg}
        super().__init__(cfg, is_train)

    def geometric_augment_que(self, que_imgs_info):
        qn, _, h, w = que_imgs_info['imgs'].shape
        imgs = que_imgs_info['imgs'].transpose([0, 2, 3, 1])
        masks = que_imgs_info['masks'][:,0,:,:]
        Ms = que_imgs_info['Ms']
        imgs_out, masks_out, Ms_out = [], [], []
        for qi in range(qn):
            scale_aug = 2 ** np.random.uniform(*self.cfg['selector_scale_range'])
            rotation_aug = np.random.uniform(*self.cfg['selector_angle_range'])
            rotation_aug = np.deg2rad(rotation_aug)

            M = transformation_offset_2d(-w/2, -h/2)
            M = transformation_compose_2d(M, transformation_rotation_2d(rotation_aug))
            M = transformation_compose_2d(M, transformation_scale_2d(scale_aug))
            M = transformation_compose_2d(M, transformation_offset_2d(w/2, h/2))

            imgs_out.append(cv2.warpAffine(imgs[qi], M, (w,h), flags=cv2.INTER_LINEAR))
            masks_out.append(cv2.warpAffine(masks[qi].astype(np.float32), M, (w,h), flags=cv2.INTER_LINEAR))
            Ms_out.append(transformation_compose_2d(Ms[qi], M))

        que_imgs_info['imgs'] = np.stack(imgs_out, 0).transpose([0,3,1,2])
        que_imgs_info['masks'] = np.stack(masks_out, 0)[:,None]
        que_imgs_info['Ms'] = np.stack(Ms_out, 0)
        return que_imgs_info

    @staticmethod
    def geometric_augment_ref(ref_imgs_in, ref_mask_in, detection_angles):
        rfn, _, h, w = ref_imgs_in.shape
        assert(h==w)
        imgs_out, masks_out = [], []
        for rfi in range(rfn):
            imgs, masks = [], []
            for angle in detection_angles:
                M = transformation_offset_2d(-h/2, -w/2)
                M = transformation_compose_2d(M, transformation_rotation_2d(np.deg2rad(angle)))  # q2r
                M = transformation_compose_2d(M, transformation_offset_2d(w/2, h/2))
                img_rotation = cv2.warpAffine(ref_imgs_in[rfi].transpose([1,2,0]), M, (w, h), flags=cv2.INTER_LINEAR) # h,w,3
                mask_rotation = cv2.warpAffine(ref_mask_in[rfi][0].astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR)
                imgs.append(img_rotation) # h,w,3
                masks.append(mask_rotation) # h,w

            imgs_out.append(np.stack(imgs,0)) # an,shape,shape,3
            masks_out.append(np.stack(masks,0)) # an,shape,shape

        imgs_out = np.stack(imgs_out,1).transpose([0,1,4,2,3]) # an,rfn,3,h,w
        masks_out = np.stack(masks_out,1)[:,:,None,:,:]        # an,rfn,1,h,w
        return imgs_out, masks_out

    @staticmethod
    def que_ref_scale_rotation_from_poses(center, ref_imgs_info, que_imgs_info):
        """
        call this function before to torch
        @param center: used in
        @param ref_imgs_info:
        @param que_imgs_info:
        @return:
            1. transformation from reference to query
               rotate(scale(ref)) = que
            2. gt_ref_ids [qn,] the ground truth reference view index for every query view
        """

        ref_poses = ref_imgs_info['poses']
        que_poses = que_imgs_info['poses']
        ref_Ks = ref_imgs_info['Ks']
        que_Ks = que_imgs_info['Ks']

        # select nearest views
        corr = compute_normalized_view_correlation(que_poses, ref_poses, center, False)
        gt_ref_ids = np.argmax(corr, 1)  # qn

        scale_diff, rotation_diff = scale_rotation_difference_from_cameras(
            ref_poses[gt_ref_ids], que_poses, ref_Ks[gt_ref_ids], que_Ks, center) # from ref to que

        ref_scales = np.asarray([transformation_decompose_scale(M) for M in ref_imgs_info['Ms'][gt_ref_ids]])
        que_scales = np.asarray([transformation_decompose_scale(M) for M in que_imgs_info['Ms']])
        ref_rotations = np.asarray([transformation_decompose_rotation(M) for M in ref_imgs_info['Ms'][gt_ref_ids]])
        que_rotations = np.asarray([transformation_decompose_rotation(M) for M in que_imgs_info['Ms']])
        scale = scale_diff * que_scales / ref_scales
        rotation = -ref_rotations + rotation_diff + que_rotations
        return scale, rotation, gt_ref_ids, corr

    @staticmethod
    def get_back_imgs(fn_list, qn,h,w):
        back_imgs = []
        for qi in range(qn):
            fn = fn_list[np.random.randint(0, len(fn_list))]
            back_img = get_background_image_coco(fn, h, w)  # h,w,3
            back_img = torch.from_numpy(color_map_forward(back_img)).permute(2, 0, 1)
            back_imgs.append(back_img)
        back_imgs = torch.stack(back_imgs, 0)  # qn,3,h,w
        return back_imgs

    def _build_real_ref_imgs_info(self, database):
        # load ref img info
        ref_ids_all = database.get_img_ids()
        res = self.cfg['resolution']
        ref_num = self.cfg['reference_num']
        angles = np.deg2rad(np.asarray(self.cfg['selector_angles']))
        ref_ids = select_reference_img_ids_fps(database, ref_ids_all, ref_num, self.is_train)
        imgs, masks, Ks, poses, Hs, ref_imgs = \
            normalize_reference_views(database, ref_ids, res, 0.05, add_rots=True, rots_list=angles)
        # imgs, masks, Ks, poses, Hs, ref_ids, ref_imgs = select_reference_views(
        #     database, ref_ids_all, ref_num, res, 0.05, self.is_train, True, angles)

        rfn = imgs.shape[0]
        M = np.concatenate([np.eye(2), np.zeros([2, 1])], 1)
        object_center = get_object_center(database)
        cens = np.asarray([project_points(object_center[None], pose, K)[0][0] for pose, K in zip(poses, Ks)],np.float32)
        ref_imgs_info = {
            'imgs': color_map_forward(imgs).transpose([0,3,1,2]),
            'masks': np.ones([rfn,1,128,128],dtype=np.float32),
            'ref_ids': [None for _ in range(rfn)],
            'Ms': np.repeat(M[None,:], rfn, 0),
            'poses': poses.astype(np.float32),
            'Ks': Ks.astype(np.float32),
            'cens': cens, # rfn, 2
        }
        ref_masks = None
        # an,rfn,h,w,3
        ref_imgs = color_map_forward(ref_imgs).transpose([0,1,4,2,3]) # an,rfn,3,h,w
        return ref_imgs_info, ref_imgs, ref_masks

    def _build_real_que_imgs_info(self, database, que_ids, center_np, ref_poses, ref_Ks, size):
        # load que imgs info
        outputs = [[] for _ in range(8)]
        for qi in range(len(que_ids)):
            img_id = que_ids[qi]
            que_img = database.get_image(img_id)
            que_pose = database.get_pose(img_id)
            que_K = database.get_K(img_id)
            que_cen = project_points(center_np[None],que_pose, que_K)[0][0]

            ref_vp_score = compute_normalized_view_correlation(que_pose[None], ref_poses, center_np, False)[0]
            gt_ref_id = np.argmax(ref_vp_score)  # qn
            scale_r2q, angle_r2q = scale_rotation_difference_from_cameras(ref_poses[gt_ref_id[None]], que_pose[None],
                                                                          ref_Ks[gt_ref_id[None]], que_K[None], center_np)
            scale_r2q, angle_r2q = scale_r2q[0], angle_r2q[0]
            if self.cfg['selector_real_aug']:
                scale_aug = 2 ** np.random.uniform(*self.cfg['selector_scale_range'])
                rotation_aug = np.deg2rad(np.random.uniform(*self.cfg['selector_angle_range']))
                que_img, M = transformation_crop(que_img, que_cen, 1/scale_r2q * scale_aug, -angle_r2q + rotation_aug, size)
                scale_r2q, angle_r2q = scale_aug, rotation_aug
            else:
                que_img, M = transformation_crop(que_img, que_cen, 1/scale_r2q, 0, size)
                scale_r2q = 1.0 # we only rescale here
            que_cen = transformation_apply_2d(M, que_cen[None])[0]

            # que_imgs, que_poses, que_Ks, que_cens, angles_r2q, scales_r2q, ref_vp_scores, gt_ref_ids
            data = [que_img, que_pose, que_K, que_cen, angle_r2q, scale_r2q, ref_vp_score, gt_ref_id]
            for output, item in zip(outputs, data):
                output.append(np.asarray(item))

        for k in range(len(outputs)):
            outputs[k] = np.stack(outputs[k], 0)
        que_imgs, que_poses, que_Ks, que_cens, angles_r2q, scales_r2q, ref_vp_scores, gt_ref_ids = outputs

        que_imgs_info = {
            'imgs': color_map_forward(que_imgs).transpose([0,3,1,2]), # qn,3,h,w
            'poses': que_poses.astype(np.float32), # qn,3,4
            'Ks': que_Ks.astype(np.float32), # qn,3,3
            'cens': que_cens.astype(np.float32), # qn,2
        }
        ref_vp_scores = ref_vp_scores.astype(np.float32) # qn, rfn
        angles_r2q = angles_r2q.astype(np.float32) # qn
        scales_r2q = scales_r2q.astype(np.float32) # qn
        gt_ref_ids = gt_ref_ids.astype(np.int64) # qn
        return que_imgs_info, angles_r2q, scales_r2q, ref_vp_scores, gt_ref_ids

    def __getitem__(self, index):
        set_seed(index,self.is_train)
        database, que_ids = self._select_query(index)
        if database.database_name.startswith('linemod'):
            object_center = get_object_center(database)
            ref_imgs_info, ref_imgs, ref_masks = self._build_real_ref_imgs_info(database)
            que_imgs_info, angles_r2q, scales_r2q, ref_vp_scores, gt_ref_ids = \
                self._build_real_que_imgs_info(database, que_ids, object_center, ref_imgs_info['poses'], ref_imgs_info['Ks'], 128)
        elif database.database_name.startswith('genmop'):
            ref_database = self.name2database[database.database_name.replace('test','ref')]
            object_center = get_object_center(database)
            ref_imgs_info, ref_imgs, ref_masks = self._build_real_ref_imgs_info(ref_database)
            que_imgs_info, angles_r2q, scales_r2q, ref_vp_scores, gt_ref_ids = \
                self._build_real_que_imgs_info(database, que_ids, object_center, ref_imgs_info['poses'], ref_imgs_info['Ks'], 128)
        else:
            ref_ids = get_ref_ids(database, self.cfg['ref_type'])
            ref_imgs_info = self._build_ref_imgs_info(database, ref_ids)
            que_imgs_info = self._build_ref_imgs_info(database, que_ids)
            ref_imgs_info.pop('ref_ids')
            que_imgs_info.pop('ref_ids')

            # add transformation for query images
            que_imgs_info = self.geometric_augment_que(que_imgs_info)

            # add transformation for reference images
            ref_imgs, ref_masks = self.geometric_augment_ref(
                ref_imgs_info['imgs'],ref_imgs_info['masks'],self.cfg['selector_angles']) # an,rfn,_,h,w

            # compute scale and rotation difference
            center = get_object_center(database)
            scales_r2q, angles_r2q, gt_ref_ids, ref_vp_scores = self.\
                que_ref_scale_rotation_from_poses(center, ref_imgs_info, que_imgs_info)
            ref_masks = torch.from_numpy(ref_masks.astype(np.float32))

        # to torch tensor
        ref_imgs = torch.from_numpy(ref_imgs.astype(np.float32))
        scales_r2q = torch.from_numpy(scales_r2q.astype(np.float32))
        angles_r2q = torch.from_numpy(angles_r2q.astype(np.float32))
        ref_vp_scores = torch.from_numpy(ref_vp_scores.astype(np.float32))
        gt_ref_ids = torch.from_numpy(gt_ref_ids.astype(np.int64))

        ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        que_imgs_info = imgs_info_to_torch(que_imgs_info)

        if database.database_name.startswith('linemod') or \
                database.database_name.startswith('genmop'):
            pass
        else:
            # add background to que imgs
            qn, _, h, w = que_imgs_info['imgs'].shape
            back_imgs = self.get_back_imgs(self.background_img_list, qn, h, w)
            que_imgs_info['imgs'] = back_imgs * (1 - que_imgs_info['masks']) + que_imgs_info['imgs'] * que_imgs_info['masks']

            # add background to ref imgs
            an, rfn, _, h, w = ref_imgs.shape
            if np.random.random() < 0.5:
                back_imgs = self.get_back_imgs(self.background_img_list, 1, h, w).unsqueeze(0)
            else:
                back_imgs = self.get_back_imgs(self.background_img_list, rfn, h, w).unsqueeze(0)
            ref_imgs = back_imgs * (1 - ref_masks) + ref_imgs * ref_masks

        # add photometric augmentation
        self._photometric_augment(que_imgs_info, 0.8)
        an, rfn, _, h, w = ref_imgs.shape
        ref_imgs = self._photometric_augment_imgs(ref_imgs.reshape(an*rfn, 3, h, w), 0.5)
        ref_imgs = ref_imgs.reshape(an, rfn, 3, h, w)
        object_center, object_vert = get_object_center(database), get_object_vert(database)
        object_center, object_vert = torch.from_numpy(object_center.astype(np.float32)), torch.from_numpy(object_vert.astype(np.float32))
        # self.check(que_imgs_info, ref_imgs, ref_vp_scores, angles_r2q, index)
        return {'ref_imgs_info': ref_imgs_info, 'que_imgs_info': que_imgs_info, 'ref_imgs': ref_imgs,
                'scales_r2q': scales_r2q, 'angles_r2q': angles_r2q, 'gt_ref_ids': gt_ref_ids, 'ref_vp_scores': ref_vp_scores,
                'object_center': object_center, 'object_vert': object_vert}

class SelectionValDataset(Dataset):
    default_cfg={
        "test_database_name": 'linemod/cat',
        "ref_database_name": 'linemod/cat',
        "test_split_type": "linemod_val",
        "ref_split_type": "linemod_val",
        "selector_ref_num": 32,
        "selector_ref_res": 128,
        'selector_angles': [-90, -45, 0, 45, 90],
    }
    def __init__(self, cfg, is_train):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__()
        assert(not is_train)
        self.test_database = parse_database_name(self.cfg['test_database_name'])
        self.ref_database = parse_database_name(self.cfg['ref_database_name'])
        ref_ids, _ = get_database_split(self.ref_database,self.cfg['ref_split_type'])
        _, self.test_ids = get_database_split(self.test_database,self.cfg['test_split_type'])

        rots = np.deg2rad(self.cfg['selector_angles'])
        ref_ids = select_reference_img_ids_fps(self.ref_database, ref_ids, self.cfg['selector_ref_num'], False)
        ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs, ref_imgs_rots = normalize_reference_views(
            self.ref_database, ref_ids, self.cfg['selector_ref_res'], 0.05, add_rots=True, rots_list=rots)
        self.ref_info={
            'poses': torch.from_numpy(ref_poses.astype(np.float32)),
            'Ks': torch.from_numpy(ref_Ks.astype(np.float32)),
            'imgs': torch.from_numpy(color_map_forward(ref_imgs)).permute(0,3,1,2)
        }
        # self.ref_pc = get_ref_point_cloud(self.ref_database).astype(np.float32)
        self.center = get_object_center(self.ref_database).astype(np.float32)
        self.res = self.cfg['selector_ref_res']
        self.ref_imgs_rots = torch.from_numpy(color_map_forward(ref_imgs_rots)).permute(0,1,4,2,3)

    def __getitem__(self, index):
        ref_imgs_info = self.ref_info.copy()
        img_id = self.test_ids[index]

        # get query information
        que_img = self.test_database.get_image(img_id)
        center_np = self.center
        que_poses = self.test_database.get_pose(img_id)[None,]
        que_Ks = self.test_database.get_K(img_id)[None,]
        que_cen = project_points(center_np[None],que_poses[0],que_Ks[0])[0][0]

        # reference information
        ref_poses = ref_imgs_info['poses'].numpy()
        ref_Ks = ref_imgs_info['Ks'].numpy()

        ref_vp_scores = compute_normalized_view_correlation(que_poses, ref_poses, center_np, False)
        gt_ref_ids = np.argmax(ref_vp_scores, 1)  # qn
        scales_r2q, angles_r2q = scale_rotation_difference_from_cameras(ref_poses[gt_ref_ids], que_poses, ref_Ks[gt_ref_ids], que_Ks, center_np)

        an, rfn, _, h, w = self.ref_imgs_rots.shape
        que_img, _ = transformation_crop(que_img, que_cen, 1/scales_r2q[0], 0, h)

        que_imgs_info = {
            'imgs': torch.from_numpy(color_map_forward(que_img)[None]).permute(0,3,1,2),
            # 'poses': torch.from_numpy(que_poses.astype(np.float32)),
            # 'Ks': torch.from_numpy(que_Ks.astype(np.float32)),
            # 'cens': torch.from_numpy(np.asarray([h/2,w/2],np.float32)[None]),
        }
        scales_r2q = torch.from_numpy(scales_r2q.astype(np.float32))
        angles_r2q = torch.from_numpy(angles_r2q.astype(np.float32))
        gt_ref_ids = torch.from_numpy(gt_ref_ids.astype(np.int64))
        ref_vp_scores = torch.from_numpy(ref_vp_scores.astype(np.float32))

        object_center, object_vert = get_object_center(self.test_database), get_object_vert(self.test_database)
        object_center, object_vert = torch.from_numpy(object_center.astype(np.float32)), torch.from_numpy(object_vert.astype(np.float32))
        # ViewSelectionGen6DDatasetV2.check(que_imgs_info,self.ref_imgs_rots,ref_vp_scores,angles_r2q,index)
        return {'que_imgs_info': que_imgs_info, 'ref_imgs_info': ref_imgs_info, 'ref_imgs': self.ref_imgs_rots,
                'gt_ref_ids': gt_ref_ids, 'scales_r2q': scales_r2q, 'angles_r2q': angles_r2q, 'ref_vp_scores': ref_vp_scores,
                "object_center": object_center, "object_vert": object_vert}

    def __len__(self):
        return len(self.test_ids)

class RefinerTrainDataset(Gen6DTrainDataset):
    refine_default_cfg={
        "batch_size": 1,
        "refine_scale_range": [-0.3, 0.3],
        "refine_rotation_range": [-15, 15],
        "refine_offset_std": 4,
        "refine_ref_num": 6,
        "refine_resolution": 128,
        "refine_view_cfg": "v0",
        "refine_ref_ids_version": "all",
    }
    def __init__(self, cfg, is_train):
        cfg = {**self.refine_default_cfg, **cfg}
        super().__init__(cfg, is_train)

    def get_view_config(self, database_name):
        gso_config={
            'select_max': 16,
            'ref_select_max': 24,
        }
        shapenet_config={
            'select_max': 24,
            'ref_select_max': 32,
        }
        linemod_config = {
            "select_max": 16,
            "ref_select_max": 32,
        }
        genmop_config = {
            "select_max": 16,
            "ref_select_max": 32,
        }

        if database_name.startswith('shapenet'):
            return shapenet_config
        elif database_name.startswith('gso'):
            return gso_config
        elif database_name.startswith('linemod'):
            return linemod_config
        elif database_name.startswith('genmop'):
            return genmop_config
        elif database_name.startswith('norm'):
            return self.get_view_config('/'.join(database_name.split('/')[1:]))
        else:
            raise NotImplementedError

    @staticmethod
    def approximate_rigid_to_similarity(pose_src, pose_tgt, K_src, K_tgt, center):
        f_tgt = (K_tgt[0, 0] + K_tgt[1, 1]) / 2
        f_src = (K_src[0, 0] + K_src[1, 1]) / 2

        cen_src = transform_points_pose(center[None], pose_src)[0] # [3]
        cen_tgt = transform_points_pose(center[None], pose_tgt)[0]

        # scale
        scale = cen_src[2] / cen_tgt[2] *  f_tgt / f_src

        # offset
        offset = (cen_tgt - cen_src)[:,None]
        offset[2,0] = 0 # note: we only consider 2D offset here
        offset = scale * offset

        # rotation
        pose = pose_compose(pose_inverse(pose_src), pose_tgt)
        rot = pose[:3,:3]

        # combine
        offset = offset + (cen_src[:,None] - scale * rot @ cen_src[:,None])
        sim = np.concatenate([scale * rot, offset],1)
        return sim

    @staticmethod
    def decomposed_transformations(pose_in, pose_sim, object_center):
        cen0 = pose_apply(pose_in, object_center)
        cen1 = pose_apply(pose_sim, cen0)
        offset = cen1 - cen0
        U, S, V = np.linalg.svd(pose_sim[:, :3])
        rotation = mat2quat(U @ V)
        scale = np.mean(np.abs(S))
        return scale, rotation, offset

    def _select_query_input_id(self, index):
        # select
        que_database, que_id = self._select_query(index)
        que_id = que_id[0]
        que_pose = que_database.get_pose(que_id)

        view_cfg = self.get_view_config(que_database.database_name)
        if que_database.database_name.startswith('gen6d'):
            ref_database = self.name2database[que_database.database_name.replace('test','ref')]
        else:
            ref_database = que_database

        input_ids = ref_database.get_img_ids()
        input_ids = np.asarray(input_ids)
        input_poses = np.stack([ref_database.get_pose(input_id) for input_id in input_ids],0).astype(np.float32)
        object_center = get_object_center(que_database)
        corr = compute_normalized_view_correlation(que_pose[None], input_poses, object_center, False)[0] # rfn
        near_idx = np.argsort(-corr)
        near_idx = near_idx[:view_cfg['select_max']]
        near_idx = near_idx[np.random.randint(0,near_idx.shape[0])]
        input_id = input_ids[near_idx]
        return que_database, ref_database, que_id, input_id

    def _get_que_imgs_info(self, que_database, ref_database, que_id, input_id, margin=0.05):
        que_img = que_database.get_image(que_id)
        que_mask = que_database.get_mask(que_id).astype(np.float32)
        que_pose = que_database.get_pose(que_id)
        que_K = que_database.get_K(que_id)
        object_center = get_object_center(que_database)
        object_diameter = get_diameter(que_database)

        # augmentation parameters
        scale_aug = 2 ** np.random.uniform(*self.cfg['refine_scale_range'])
        angle_aug = np.deg2rad(np.random.uniform(*self.cfg['refine_rotation_range']))
        offset_aug = np.random.normal(0, self.cfg['refine_offset_std'], 2).astype(np.float32)

        # we need to rescale the input image to the target size
        size = self.cfg['refine_resolution']
        input_pose, input_K = ref_database.get_pose(input_id), ref_database.get_K(input_id)
        is_synthetic = False
        if not is_synthetic:
            # compute the scale to correct input to a fixed size
            # let the input pose and input K look at the obejct
            input_dist = np.linalg.norm(pose_inverse(input_pose)[:,3] - object_center[None,], 2)
            input_rot_look, input_focal_look = let_me_look_at(input_pose, input_K, object_center)
            input_pose = pose_compose(input_pose, np.concatenate([input_rot_look,np.zeros([3,1])],1)) # new input pose
            input_focal_new = size * (1 - margin) / object_diameter * input_dist
            input_K = np.diag([input_focal_new,input_focal_new,1.0])
            input_K[:,2] = np.asarray([size/2, size/2, 1.0]) # new input K

            scale_diff, angle_diff = scale_rotation_difference_from_cameras(
                input_pose[None], que_pose[None], input_K[None], que_K[None], object_center)
            scale_diff, angle_diff = scale_diff[0], angle_diff[0] # input to query

            # rotation
            que_cen = project_points(object_center, que_pose, que_K)[0][0]
            R_new, f_new = let_me_look_at_2d(que_cen + offset_aug, que_K)
            angle = angle_aug - angle_diff
            R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
            R_new = R_z @ R_new

            # scale
            f_new = f_new * scale_aug / scale_diff
            que_K_warp = np.asarray([[f_new,0,size/2],[0,f_new,size/2],[0,0,1]],np.float32)

            # warp image
            H = que_K_warp @ R_new @ np.linalg.inv(que_K)
            que_img_warp = cv2.warpPerspective(que_img, H, (size, size), flags=cv2.INTER_LINEAR)
            que_mask_warp = cv2.warpPerspective(que_mask.astype(np.float32), H, (size, size), flags=cv2.INTER_LINEAR)

            # compute ground-truth pose of the warped image and similarity pose
            pose_rect = np.concatenate([R_new,np.zeros([3,1])],1).astype(np.float32)
            que_pose_warp = pose_compose(que_pose, pose_rect)
            # todo: the approximation is not accurate, if the the object is close to the camera. so we use different codes for synthetic data
            poses_sim_in_to_warp = self.approximate_rigid_to_similarity(input_pose, que_pose_warp, input_K, que_K_warp, object_center)
        else:
            raise NotImplementedError
            # scale_diff, angle_diff = scale_rotation_difference_from_cameras(
            #     input_pose[None], que_pose[None], input_K[None], que_K[None], object_center)
            # scale_diff, angle_diff = scale_diff[0], angle_diff[0]
            #
            # que_cen, que_depth = project_points(object_center, que_pose, que_K)
            # que_cen, que_depth = que_cen[0], que_depth[0]
            # input_cen, input_depth = project_points(object_center, input_pose, input_K)
            # input_cen, input_depth = input_cen[0], input_depth[0]
            #
            # M = transformation_offset_2d(-que_cen[0], -que_cen[1])
            # M = transformation_compose_2d(M, transformation_scale_2d(1 / scale_diff))
            # M = transformation_compose_2d(M, transformation_rotation_2d(-angle_diff))
            # M = transformation_compose_2d(M, transformation_offset_2d(offset_aug[0], offset_aug[1]))
            # M = transformation_compose_2d(M, transformation_scale_2d(scale_aug))
            # M = transformation_compose_2d(M, transformation_rotation_2d(angle_aug))
            # M = transformation_compose_2d(M, transformation_offset_2d(input_cen[0], input_cen[1]))
            #
            # que_img_warp = cv2.warpAffine(que_img, M, (size, size), flags=cv2.INTER_LINEAR)
            # que_mask_warp = cv2.warpAffine(que_mask.astype(np.float32), M, (size, size), flags=cv2.INTER_LINEAR)
            # H = np.identity(3)
            # H[:2,:3] = M
            #
            # # compute the pose similarity transformation
            # # rotation
            # angle = angle_aug - angle_diff
            # R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
            # R0, R1 = que_pose[:, :3], input_pose[:, :3]
            # rotation_i2q = R_z @ R0 @ R1.T
            #
            # # scale
            # scale_i2q = scale_aug
            #
            # # offset
            # que_cen_ = transformation_apply_2d(M, que_cen[None])[0]
            # offset2d_i2q = que_cen_ - input_cen
            # input_f = np.mean(np.diag(input_K)[:2])
            # offset2d_i2q = offset2d_i2q / input_f * input_depth
            # offset2d_i2q = np.append(offset2d_i2q, 0).astype(np.float32)
            #
            # ref_obj_cen = pose_apply(input_pose, object_center)
            # offset2d_i2q = offset2d_i2q + ref_obj_cen - scale_i2q * rotation_i2q @ ref_obj_cen
            # poses_sim_in_to_warp = np.concatenate([scale_i2q * rotation_i2q, offset2d_i2q[:, None]], 1).astype(np.float32)

        que_imgs_info={
            # warp pose info
            'imgs': color_map_forward(que_img_warp).transpose([2, 0, 1]),  # h,w,3
            'masks': que_mask_warp[None].astype(np.float32),  # 1,h,w
            "Ks": que_K_warp.astype(np.float32),
            "poses": que_pose_warp.astype(np.float32),

            # input pose info
            'Hs': H.astype(np.float32), # 3,3
            'Ks_in': input_K.astype(np.float32), # 3,3
            'poses_in': input_pose.astype(np.float32), # 3,4
            'poses_sim_in_to_que': np.asarray(poses_sim_in_to_warp,np.float32), # 3,4
        }
        # compute decomposed transformations
        scale, rotation, offset = self.decomposed_transformations(input_pose, poses_sim_in_to_warp, object_center)
        return que_imgs_info, scale, rotation, offset

    def _get_ref_imgs_info(self, database, input_pose, input_K, is_synthetic, margin=0.05):
        if self.cfg['refine_ref_ids_version']=='all':
            img_ids = np.asarray(database.get_img_ids())
        elif self.cfg['refine_ref_ids_version']=='fps':
            img_ids = select_reference_img_ids_fps(database, database.get_img_ids(), 128, self.is_train)
        else:
            raise NotImplementedError
        ref_poses_all = np.asarray([database.get_pose(ref_id) for ref_id in img_ids])
        view_cfg = self.get_view_config(database.database_name)

        # select reference ids from input pose
        object_center = get_object_center(database)
        corr = compute_normalized_view_correlation(input_pose[None], ref_poses_all, object_center, False)
        ref_idxs = np.argsort(-corr[0])
        ref_idxs = ref_idxs[:view_cfg['ref_select_max']]
        np.random.shuffle(ref_idxs)
        ref_idxs = ref_idxs[:self.cfg['refine_ref_num']]
        ref_ids = img_ids[ref_idxs]

        size = self.cfg['refine_resolution']
        if is_synthetic:
            raise NotImplementedError
            # ref_imgs = [database.get_image(ref_id) for ref_id in ref_ids]
            # ref_masks = [database.get_mask(ref_id) for ref_id in ref_ids]
            # ref_Ks = [database.get_K(ref_id) for ref_id in ref_ids]
            # ref_poses = [database.get_pose(ref_id) for ref_id in ref_ids]
            # ref_poses, ref_Ks, ref_imgs, ref_masks = CostVolumeRefineEvalDataset.rectify_inplane_rotation(
            #     input_pose, input_K, object_center, ref_poses, ref_Ks, ref_imgs, ref_masks)
        else:
            # if it is a real database, we need to re-scale them to the same size.
            ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs = normalize_reference_views(
                database, ref_ids, size, margin, True, input_pose, input_K)

        ref_imgs_info={
            'imgs': color_map_forward(np.stack(ref_imgs,0)).transpose([0,3,1,2]), # rfn,3,h,w
            'masks': np.stack(ref_masks, 0).astype(np.float32)[:,None,:,:], # rfn,1,h,w
            'poses': np.stack(ref_poses, 0).astype(np.float32),
            'Ks': np.stack(ref_Ks, 0).astype(np.float32),
        }
        return ref_imgs_info

    @staticmethod
    def add_ref_background(ref_imgs, ref_masks, background_img_list):
        """
        @param ref_imgs: rfn,3,h,w
        @param ref_masks: rfn,1,h,w
        @param background_img_list:
        @return:
        """
        same_background_prob = 0.4
        if np.random.random()<0.95:
            rfn, _, h, w = ref_imgs.shape
            if np.random.random() < same_background_prob:
                fn = background_img_list[np.random.randint(0, len(background_img_list))]
                back_imgs = get_background_image_coco(fn, h, w)
                back_imgs = color_map_forward(back_imgs).transpose([2,0,1])[None,:]
            else:
                rfn = ref_imgs.shape[0]
                back_imgs = []
                for rfi in range(rfn):
                    fn = background_img_list[np.random.randint(0, len(background_img_list))]
                    back_img = get_background_image_coco(fn, h, w)
                    back_img = color_map_forward(back_img).transpose([2,0,1])
                    back_imgs.append(back_img)
                back_imgs = np.stack(back_imgs, 0)
            ref_imgs = ref_imgs * ref_masks + back_imgs * (1 - ref_masks)
        return ref_imgs

    @staticmethod
    def add_que_background(que_img, que_mask, background_img_list):
        """
        @param que_img:  [3,h,w]
        @param que_mask: [1,h,w]
        @param background_img_list:
        @return:
        """
        _, h, w = que_img.shape
        if np.random.random() < 0.95:
            fn = background_img_list[np.random.randint(0, len(background_img_list))]
            back_img = get_background_image_coco(fn, h, w)
            back_img = color_map_forward(back_img).transpose([2,0,1])
            que_img = que_img * que_mask + back_img * (1 - que_mask)
        return que_img

    def __getitem__(self, index):
        set_seed(index, self.is_train)
        que_database, ref_database, que_id, input_id = self._select_query_input_id(index)
        is_synthetic = que_database.database_name.startswith('gso') or que_database.database_name.startswith('shapenet')
        que_database = NormalizedDatabase(que_database)
        ref_database = NormalizedDatabase(ref_database)

        que_imgs_info, scale, rotation, offset = self._get_que_imgs_info(que_database, ref_database, que_id, input_id)
        input_pose, input_K = que_imgs_info['poses_in'], que_imgs_info['Ks_in']
        ref_imgs_info = self._get_ref_imgs_info(ref_database, input_pose, input_K, False)

        object_center = get_object_center(que_database)
        object_center = torch.from_numpy(object_center.astype(np.float32))
        rotation = torch.from_numpy(np.asarray(rotation,np.float32))
        scale = torch.from_numpy(np.asarray(scale,np.float32))
        offset = torch.from_numpy(np.asarray(offset,np.float32))

        # add background
        if is_synthetic:
            ref_imgs_info['imgs'] = self.add_ref_background(ref_imgs_info['imgs'],ref_imgs_info['masks'],self.background_img_list)
            que_imgs_info['imgs'] = self.add_que_background(que_imgs_info['imgs'],que_imgs_info['masks'],self.background_img_list)

        # pc = get_ref_point_cloud(que_database)
        # CostVolumeRefineDataset.check(que_imgs_info, ref_imgs_info, pc, f'data/vis_val/{index}.jpg')

        que_imgs_info = imgs_info_to_torch(que_imgs_info)
        ref_imgs_info = imgs_info_to_torch(ref_imgs_info)

        # add dataaugmentation
        self._photometric_augment(que_imgs_info, 0.8)
        self._photometric_augment(ref_imgs_info, 0.8)
        return {'que_imgs_info': que_imgs_info, 'ref_imgs_info': ref_imgs_info, 'object_center': object_center,
                'rotation': rotation, 'scale': scale, 'offset': offset}

class RefinerValDataset(Dataset):
    default_cfg={
        'ref_database_name': 'linemod/cat',
        'ref_split_type': 'linemod_test',
        'test_database_name': 'linemod/cat',
        'test_split_type': 'linemod_test',

        "selector_name": "selector_train",
        "detector_name": "detector_train",
        "refine_ref_num": 5,
        "refine_resolution": 128,
        "refine_even_ref_views": True,
    }
    def __init__(self, cfg, is_train):
        self.cfg={**self.default_cfg, **cfg}
        self.test_database = parse_database_name(self.cfg['test_database_name'])
        self.ref_database = parse_database_name(self.cfg['ref_database_name'])
        _, self.test_ids = get_database_split(self.test_database, self.cfg['test_split_type'])
        self.ref_ids, _ = get_database_split(self.ref_database, self.cfg['ref_split_type'])
        self.ref_ids, self.test_ids = np.asarray(self.ref_ids), np.asarray(self.test_ids)

        self.img_id2det_info = read_pickle(f'data/val/det/{self.test_database.database_name}/{self.cfg["detector_name"]}.pkl')
        self.img_id2sel_info = read_pickle(f'data/val/sel/{self.test_database.database_name}/{self.cfg["detector_name"]}-{self.cfg["selector_name"]}.pkl')

    def __getitem__(self, index):
        que_id = self.test_ids[index]
        test_database = NormalizedDatabase(self.test_database)
        ref_database = NormalizedDatabase(self.ref_database)
        que_img = test_database.get_image(que_id)
        que_mask = test_database.get_mask(que_id)
        que_pose = test_database.get_pose(que_id)
        que_K = test_database.get_K(que_id)
        center = get_object_center(ref_database)
        res = self.cfg['refine_resolution']

        det_position, det_scale_r2q, _ = self.img_id2det_info[que_id]
        sel_angle_r2q, sel_pose, sel_K = self.img_id2sel_info[que_id]
        # remember to normalize the pose !!!
        sel_pose = normalize_pose(sel_pose, test_database.scale, test_database.offset)

        que_img_warp, que_K_warp, que_pose_warp, que_pose_rect, H = look_at_crop(
            que_img, que_K, que_pose, det_position, -sel_angle_r2q, 1/det_scale_r2q, res, res)
        que_mask_warp = cv2.warpPerspective(que_mask.astype(np.float32), H, (res, res), flags=cv2.INTER_LINEAR)
        poses_sim_in_to_warp = RefinerTrainDataset.approximate_rigid_to_similarity(
            sel_pose, que_pose_warp, sel_K, que_K_warp, center)

        pose_in_raw = estimate_pose_from_similarity_transform_compose(
            det_position, det_scale_r2q, sel_angle_r2q, sel_pose, sel_K, que_K, center)

        que_imgs_info={
            # warp pose info
            'imgs': color_map_forward(que_img_warp).transpose([2,0,1]),  # 3,h,w
            'masks': que_mask_warp.astype(np.float32),  # 1,h,w
            "Ks": que_K_warp.astype(np.float32),
            "poses": que_pose_warp.astype(np.float32),
            "poses_rect": np.asarray(que_pose_rect, np.float32),

            # input pose info
            'Hs': H.astype(np.float32), # 3,3
            'Ks_in': sel_K.astype(np.float32), # 3,3
            'poses_in': sel_pose.astype(np.float32), # 3,4
            "poses_sim_in_to_que": poses_sim_in_to_warp.astype(np.float32),  # 3,4

            # original image and pose info
            'imgs_raw': color_map_forward(que_img).transpose([2,0,1]),  # 3,h,w
            'masks_raw': que_mask[None].astype(np.float32),  # 1,h,w
            'poses_raw': que_pose.astype(np.float32),  # 3,4
            'Ks_raw': que_K.astype(np.float32),  # 3,3
            'pose_in_raw': pose_in_raw.astype(np.float32)
        }

        scale, rotation, offset = RefinerTrainDataset.decomposed_transformations(sel_pose, poses_sim_in_to_warp, center)
        rotation = torch.from_numpy(np.asarray(rotation,np.float32))
        scale = torch.from_numpy(np.asarray(scale,np.float32))
        offset = torch.from_numpy(np.asarray(offset,np.float32))

        ref_ids = select_reference_img_ids_refinement(
            ref_database, center, self.ref_ids, sel_pose, self.cfg['refine_ref_num'], self.cfg['refine_even_ref_views'])

        size = self.cfg['refine_resolution']
        margin = 0.05
        ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs = normalize_reference_views(ref_database, ref_ids, size, margin, True, sel_pose, sel_K)
        # ref_poses, ref_Ks, ref_imgs, ref_masks = CostVolumeRefineEvalDataset.rectify_inplane_rotation(
        #     sel_pose, sel_K, object_center, ref_poses, ref_Ks, ref_imgs, ref_masks)
        ref_imgs_info={
            'imgs': color_map_forward(np.stack(ref_imgs,0)).transpose([0,3,1,2]), # rfn,3,h,w
            'masks': np.stack(ref_masks, 0).astype(np.float32)[:,None,:,:], # rfn,1,h,w
            'poses': np.stack(ref_poses, 0).astype(np.float32),
            'Ks': np.stack(ref_Ks, 0).astype(np.float32),
        }

        diameter = np.asarray(get_diameter(test_database),np.float32)
        points = get_ref_point_cloud(test_database).astype(np.float32)

        que_imgs_info = imgs_info_to_torch(que_imgs_info)
        ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        diameter = torch.from_numpy(diameter)
        points = torch.from_numpy(points)
        center = torch.from_numpy(center)

        que_img_raw = test_database.get_image(que_id)
        que_img_raw = torch.from_numpy(color_map_forward(que_img_raw)).permute(2,0,1).unsqueeze(0)
        return {'que_imgs_info': que_imgs_info, 'ref_imgs_info':ref_imgs_info, 'object_diameter': diameter, 'object_points': points, 'object_center': center,
                'que_img_raw': que_img_raw, 'que_id': que_id, 'database_name': self.test_database.database_name, 'rotation': rotation, 'scale': scale, 'offset': offset}

    def __len__(self):
        return len(self.test_ids)

name2dataset={
    'det_train': DetectionTrainDataset,
    'det_val': DetectionValDataset,
    'sel_train': SelectionTrainDataset,
    'sel_val': SelectionValDataset,
    'ref_train': RefinerTrainDataset,
    'ref_val': RefinerValDataset,
}