import abc
import glob
import gzip
from pathlib import Path
import random
from typing import List

import numpy as np
import cv2
import os

import plyfile
from PIL import Image
from skimage.io import imread, imsave
from tqdm import tqdm

from utils.base_utils import read_pickle, save_pickle, sample_fps_points, pose_compose, load_point_cloud
from utils.read_write_model import read_model

SUN_IMAGE_ROOT = 'data/SUN2012pascalformat/JPEGImages'
SUN_IMAGE_ROOT_128 = 'data/SUN2012pascalformat/JPEGImages_128'
SUN_IMAGE_ROOT_256 = 'data/SUN2012pascalformat/JPEGImages_256'
SUN_IMAGE_ROOT_512 = 'data/SUN2012pascalformat/JPEGImages_512'
SUN_IMAGE_ROOT_32 = 'data/SUN2012pascalformat/JPEGImages_64'
def get_SUN397_image_fn_list():
    if Path('data/SUN397_list.pkl').exists():
        return read_pickle('data/SUN397_list.pkl')
    img_list = os.listdir(SUN_IMAGE_ROOT)
    img_list = [img for img in img_list if img.endswith('.jpg')]
    save_pickle(img_list, 'data/SUN397_list.pkl')
    return img_list

COCO_IMAGE_ROOT = 'data/coco/train2017'
def get_COCO_image_fn_list():
    if Path('data/COCO_list.pkl').exists():
        return read_pickle('data/COCO_list.pkl')
    img_list = os.listdir(COCO_IMAGE_ROOT)
    img_list = [img for img in img_list if img.endswith('.jpg')]
    save_pickle(img_list, 'data/COCO_list.pkl')
    return img_list

def mask2bbox(mask):
    if np.sum(mask)==0:
        return np.asarray([0, 0, 0, 0],np.float32)
    ys, xs = np.nonzero(mask)
    x_min = np.min(xs)
    y_min = np.min(ys)
    x_max = np.max(xs)
    y_max = np.max(ys)
    return np.asarray([x_min, y_min, x_max - x_min, y_max - y_min], np.int32)

class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self):
        pass

    @abc.abstractmethod
    def get_mask(self,img_id):
        pass

LINEMOD_ROOT='data/LINEMOD'
class LINEMODDatabase(BaseDatabase):
    K=np.array([[572.4114, 0., 325.2611],
               [0., 573.57043, 242.04899],
               [0., 0., 1.]], dtype=np.float32)
    def __init__(self, database_name):
        super().__init__(database_name)
        _, self.model_name = database_name.split('/')
        self.img_ids = [str(k) for k in range(len(os.listdir(f'{LINEMOD_ROOT}/{self.model_name}/JPEGImages')))]
        self.model = self.get_ply_model().astype(np.float32)
        self.object_center = np.zeros(3,dtype=np.float32)
        self.object_vert = np.asarray([0,0,1],np.float32)
        self.img_id2depth_range = {}
        self.img_id2pose = {}

    def get_ply_model(self):
        fn = Path(f'{LINEMOD_ROOT}/{self.model_name}/{self.model_name}.pkl')
        if fn.exists(): return read_pickle(str(fn))
        ply = plyfile.PlyData.read(f'{LINEMOD_ROOT}/{self.model_name}/{self.model_name}.ply')
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        if model.shape[0]>4096:
            idxs = np.arange(model.shape[0])
            np.random.shuffle(idxs)
            model = model[idxs[:4096]]
        save_pickle(model, str(fn))
        return model

    def get_image(self, img_id):
        return imread(f'{LINEMOD_ROOT}/{self.model_name}/JPEGImages/{int(img_id):06}.jpg')

    def get_K(self, img_id):
        return np.copy(self.K)

    def get_pose(self, img_id):
        if img_id in self.img_id2pose:
            return self.img_id2pose[img_id]
        else:
            pose = np.load(f'{LINEMOD_ROOT}/{self.model_name}/pose/pose{int(img_id)}.npy')
            self.img_id2pose[img_id] = pose
            return pose

    def get_img_ids(self):
        return self.img_ids.copy()

    def get_mask(self, img_id):
        return np.sum(imread(f'{LINEMOD_ROOT}/{self.model_name}/mask/{int(img_id):04}.png'),-1)>0

GenMOP_ROOT='data/GenMOP'

genmop_meta_info={
    'cup': {'gravity': np.asarray([-0.0893124,-0.399691,-0.912288]), 'forward': np.asarray([-0.009871,0.693020,-0.308549],np.float32)},
    'tformer': {'gravity': np.asarray([-0.0734401,-0.633415,-0.77032]), 'forward': np.asarray([-0.121561, -0.249061, 0.211048],np.float32)},
    'chair': {'gravity': np.asarray((0.111445, -0.373825, -0.920779),np.float32), 'forward': np.asarray([0.788313,-0.139603,0.156288],np.float32)},
    'knife': {'gravity': np.asarray((-0.0768299, -0.257446, -0.963234),np.float32), 'forward': np.asarray([0.954157,0.401808,-0.285027],np.float32)},
    'love': {'gravity': np.asarray((0.131457, -0.328559, -0.93529),np.float32), 'forward': np.asarray([-0.045739,-1.437427,0.497225],np.float32)},
    'plug_cn': {'gravity': np.asarray((-0.0267497, -0.406514, -0.913253),np.float32), 'forward': np.asarray([-0.172773,-0.441210,0.216283],np.float32)},
    'plug_en': {'gravity': np.asarray((0.0668682, -0.296538, -0.952677),np.float32), 'forward': np.asarray([0.229183,-0.923874,0.296636],np.float32)},
    'miffy': {'gravity': np.asarray((-0.153506, -0.35346, -0.922769),np.float32), 'forward': np.asarray([-0.584448,-1.111544,0.490026],np.float32)},
    'scissors': {'gravity': np.asarray((-0.129767, -0.433414, -0.891803),np.float32), 'forward': np.asarray([1.899760,0.418542,-0.473156],np.float32)},
    'piggy': {'gravity': np.asarray((-0.122392, -0.344009, -0.930955), np.float32), 'forward': np.asarray([0.079012,1.441836,-0.524981], np.float32)},
}
class GenMOPMetaInfoWrapper:
    def __init__(self, object_name):
        self.object_name = object_name
        self.gravity = genmop_meta_info[self.object_name]['gravity']
        self.forward = genmop_meta_info[self.object_name]['forward']
        self.object_point_cloud = load_point_cloud(f'{GenMOP_ROOT}/{self.object_name}-ref/object_point_cloud.ply')

        # rotate
        self.rotation = self.compute_rotation(self.gravity, self.forward)
        self.object_point_cloud = (self.object_point_cloud @ self.rotation.T)

        # scale
        self.scale_ratio = self.compute_normalized_ratio(self.object_point_cloud)
        self.object_point_cloud = self.object_point_cloud * self.scale_ratio

        min_pt = np.min(self.object_point_cloud,0)
        max_pt = np.max(self.object_point_cloud,0)
        self.center = (max_pt + min_pt)/2

        test_fn = f'{GenMOP_ROOT}/{self.object_name}-ref/test-object_point_cloud.ply'
        if Path(test_fn).exists():
            self.test_object_point_cloud = load_point_cloud(test_fn)

    @staticmethod
    def compute_normalized_ratio(pc):
        min_pt = np.min(pc,0)
        max_pt = np.max(pc,0)
        dist = np.linalg.norm(max_pt - min_pt)
        scale_ratio = 2.0 / dist
        return scale_ratio

    def normalize_pose(self, pose):
        R = pose[:3,:3]
        t = pose[:3,3:]
        R = R @ self.rotation.T
        t = self.scale_ratio * t
        return np.concatenate([R,t], 1).astype(np.float32)

    @staticmethod
    def compute_rotation(vert, forward):
        y = np.cross(vert, forward)
        x = np.cross(y, vert)

        vert = vert/np.linalg.norm(vert)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        R = np.stack([x, y, vert], 0)
        return R


class GenMOPDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)

        _, seq_name = database_name.split('/') # genmop/object_name-test or genmop/object_name-ref

        # get image filenames
        self.seq_name = seq_name
        self.root = Path(GenMOP_ROOT) / self.seq_name
        img_fns_cache = self.root / 'images_fn_cache.pkl'
        self.img_fns = read_pickle(str(img_fns_cache))

        # parse colmap project
        cameras, images, points3d = read_model(f'{GenMOP_ROOT}/{seq_name}/colmap-all/colmap_default-colmap_default/sparse/0')
        img_id2db_id = {v.name[:-4]:k for k, v in images.items()}
        self.poses, self.Ks = {}, {}
        self.img_ids = [str(k) for k in range(len(self.img_fns))]
        for img_id in self.img_ids:
            if img_id not in img_id2db_id: continue
            db_id = img_id2db_id[img_id]
            R = images[db_id].qvec2rotmat()
            t = images[db_id].tvec
            pose = np.concatenate([R,t[:,None]],1).astype(np.float32)
            self.poses[img_id]=pose

            cam_id = images[db_id].camera_id
            f, cx, cy, _ = cameras[cam_id].params
            self.Ks[img_id] = np.asarray([[f,0,cx], [0,f,cy], [0,0,1]],np.float32)

        # align test sequence to the reference sequence
        object_name, database_type = seq_name.split('-')
        if database_type=='test':
            scale_ratio, transfer_pose = read_pickle(f'{GenMOP_ROOT}/{seq_name}/align.pkl')
            for img_id in self.get_img_ids():
                pose = self.poses[img_id]
                pose_new = pose_compose(transfer_pose, pose)
                pose_new[:, 3:] *= scale_ratio
                self.poses[img_id] = pose_new

        # normalize object poses by meta info: rotate and scale not offset
        self.meta_info = GenMOPMetaInfoWrapper(object_name)
        self.poses = {img_id: self.meta_info.normalize_pose(self.poses[img_id]) for img_id in self.get_img_ids()}

    def get_image(self, img_id):
        return imread(str(self.root / 'images' / self.img_fns[int(img_id)]))

    def get_K(self, img_id):
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids

    def get_mask(self, img_id):
        # dummy mask
        img = self.get_image(img_id)
        h, w = img.shape[:2]
        return np.ones([h,w],np.bool)

def parse_database_name(database_name:str)->BaseDatabase:
    name2database={
        'linemod': LINEMODDatabase,
        'genmop': GenMOPDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name)
    else:
        raise NotImplementedError

def get_database_split(database, split_name):
    if split_name.startswith('linemod'): # linemod_test or linemod_val
        assert(database.database_name.startswith('linemod'))
        model_name = database.database_name.split('/')[1]
        lines = np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/test.txt",dtype=np.str).tolist()
        que_ids, ref_ids = [], []
        for line in lines: que_ids.append(str(int(line.split('/')[-1].split('.')[0])))
        if split_name=='linemod_val': que_ids = que_ids[::10]
        lines = np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/train.txt", dtype=np.str).tolist()
        for line in lines: ref_ids.append(str(int(line.split('/')[-1].split('.')[0])))
    elif split_name=='all':
        ref_ids = que_ids = database.get_img_ids()
    else:
        raise NotImplementedError
    return ref_ids, que_ids

def get_ref_point_cloud(database):
    if isinstance(database, LINEMODDatabase):
        ref_point_cloud = database.model
    elif isinstance(database, GenMOPDatabase):
        ref_point_cloud = database.meta_info.object_point_cloud
    else:
        raise NotImplementedError
    return ref_point_cloud

def get_diameter(database):
    if isinstance(database, LINEMODDatabase):
        model_name = database.database_name.split('/')[-1]
        return np.loadtxt(f"{LINEMOD_ROOT}/{model_name}/distance.txt") / 100
    elif isinstance(database, GenMOPDatabase):
        return 2.0 # we already align and scale it
    elif isinstance(database, NormalizedDatabase):
        return 2.0
    else:
        raise NotImplementedError

def get_object_center(database):
    if isinstance(database, LINEMODDatabase):
        return database.object_center
    elif isinstance(database, GenMOPDatabase):
        return database.meta_info.center
    elif isinstance(database, NormalizedDatabase):
        return np.zeros(3,dtype=np.float32)
    else:
        raise NotImplementedError

def get_object_vert(database):
    if isinstance(database, LINEMODDatabase):
        return database.object_vert
    elif isinstance(database, GenMOPDatabase):
        return np.asarray([0,0,1], np.float32)
    else:
        raise NotImplementedError

def normalize_pose(pose, scale, offset):
    # x_obj_new = x_obj * scale + offset
    R = pose[:3, :3]
    t = pose[:3, 3]
    t_ = R @ -offset + scale * t
    return np.concatenate([R, t_[:,None]], -1).astype(np.float32)

def denormalize_pose(pose, scale, offset):
    R = pose[:3,:3]
    t = pose[:3, 3]
    t = R @ offset / scale + t/scale
    return np.concatenate([R, t[:, None]], -1).astype(np.float32)

class NormalizedDatabase(BaseDatabase):
    def get_image(self, img_id):
        return self.database.get_image(img_id)

    def get_K(self, img_id):
        return self.database.get_K(img_id)

    def get_pose(self, img_id):
        pose = self.database.get_pose(img_id)
        return normalize_pose(pose, self.scale, self.offset)

    def get_img_ids(self, check_depth_exist=False):
        return self.database.get_img_ids()

    def get_mask(self, img_id):
        return self.database.get_mask(img_id)

    def __init__(self, database: BaseDatabase):
        super().__init__("norm/" + database.database_name)
        self.database = database
        center = get_object_center(self.database)
        diameter = get_diameter(self.database)

        self.scale = 2/diameter
        self.offset = - self.scale * center

        # self.diameter = 2.0
        # self.center = np.zeros([3],np.float32)
        # self.vert = get_object_vert(self.database)