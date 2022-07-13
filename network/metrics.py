import torch

from pathlib import Path

import numpy as np
from skimage.io import imsave
from transforms3d.axangles import mat2axangle
from transforms3d.quaternions import quat2mat

from network.loss import Loss
from utils.base_utils import color_map_backward, transformation_crop, pose_apply, pose_compose, pose_inverse, \
    project_points
from utils.bbox_utils import parse_bbox_from_scale_offset, bboxes_iou, lthw_to_ltrb
from utils.draw_utils import draw_bbox, concat_images_list, pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pose_sim_to_pose_rigid, compute_pose_errors


class VisualizeBBoxScale(Loss):
    default_cfg={
        'output_interval': 250,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([])
        self.count_num=0

    def __call__(self, data_pr, data_gt, step, **kwargs):
        data_index=kwargs['data_index']
        model_name=kwargs['model_name']
        output_root = kwargs['output_root'] if 'output_root' in kwargs else 'data/vis'

        b, _, hr, wr = data_gt['ref_imgs_info']['imgs'].shape
        que_select_id = data_pr['que_select_id'][0].cpu().numpy() # 3
        scale_pr = data_pr['select_pr_scale'].detach().cpu().numpy()[0,0]
        offset_pr = data_pr['select_pr_offset'].detach().cpu().numpy()[0]
        pool_ratio = data_pr['pool_ratio']
        ref_shape=(hr,wr)
        bbox_pr = parse_bbox_from_scale_offset(que_select_id, scale_pr, offset_pr, pool_ratio, ref_shape)

        center = data_gt['que_imgs_info']['cens'][0].cpu().numpy()
        scale_gt = data_gt['scale_diff'].cpu().numpy()[0]
        h_gt, w_gt = hr * scale_gt, wr * scale_gt
        # center = bbox[:2] + bbox[2:] / 2
        bbox_gt = np.asarray([center[0] - w_gt / 2, center[1] - h_gt / 2, w_gt, h_gt])

        iou = bboxes_iou(lthw_to_ltrb(bbox_gt[None],False),lthw_to_ltrb(bbox_pr[None],False),False)
        if data_index % self.cfg['output_interval'] != 0:
            return {'iou': iou}
        que_imgs = data_gt['que_imgs_info']['imgs']
        que_imgs = color_map_backward(que_imgs.permute(0, 2, 3, 1).cpu().numpy())
        que_img = que_imgs[0]
        ref_imgs = data_gt['ref_imgs_info']['imgs']
        ref_imgs = color_map_backward(ref_imgs.permute(0, 2, 3, 1).cpu().numpy())
        rfn, hr, wr, _ = ref_imgs.shape
        que_img = draw_bbox(que_img, bbox_pr, color=(0,0,255))
        que_img = draw_bbox(que_img, bbox_gt)
        Path(f'{output_root}/{model_name}').mkdir(exist_ok=True,parents=True)
        imsave(f'{output_root}/{model_name}/{step}-{data_index}-bbox.jpg', que_img)
        return {'iou': iou}


class VisualizeSelector(Loss):
    default_cfg={}
    def __init__(self,cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        data_index=kwargs['data_index']
        model_name=kwargs['model_name']
        output_root = kwargs['output_root'] if 'output_root' in kwargs else 'data/vis'

        outputs = {}
        logits = data_pr['ref_vp_logits'] # qn,rfn
        order_pr = torch.argsort(-logits, 1) # qn,rfn

        scores_gt = data_gt['ref_vp_scores']
        order_gt = torch.argsort(-scores_gt, 1)  # qn,rfn

        order_pr, order_gt = order_pr.cpu().numpy(), order_gt.cpu().numpy()
        order_pr_min = order_pr[:,:1]
        mask1 = np.sum(order_pr_min == order_gt[:,:1], 1).astype(np.float32)
        mask3 = np.sum(order_pr_min == order_gt[:,:3], 1).astype(np.float32)
        mask5 = np.sum(order_pr_min == order_gt[:,:5], 1).astype(np.float32)
        outputs['sel_acc_1'] = mask1
        outputs['sel_acc_3'] = mask3
        outputs['sel_acc_5'] = mask5

        angles_pr = data_pr['angles_pr'].cpu().numpy()*np.pi/2 # qn,rfn
        angles_gt = data_gt['angles_r2q'].cpu().numpy() # qn,
        gt_ref_ids = data_gt['gt_ref_ids'].cpu().numpy() # qn
        angles_pr_ = angles_pr[np.arange(gt_ref_ids.shape[0]),gt_ref_ids]
        angles_diff = angles_pr_ - angles_gt # qn
        angles_diff = np.abs(np.rad2deg(angles_diff))
        angle5 = (angles_diff < 5).astype(np.float32)
        angle15 = (angles_diff < 15).astype(np.float32)
        angle30 = (angles_diff < 30).astype(np.float32)
        outputs['sel_ang_5'] = angle5
        outputs['sel_ang_15'] = angle15
        outputs['sel_ang_30'] = angle30
        outputs['angles_diff'] = angles_diff

        if data_index % self.cfg['output_interval']!=0:
            return outputs

        # visualize selected viewpoints and regressed rotations
        ref_imgs = data_gt['ref_imgs'] # an,rfn,3,h,w
        que_imgs = data_gt['que_imgs_info']['imgs'] # qn,3,h,w
        ref_imgs = color_map_backward(ref_imgs.cpu().numpy()).transpose([0,1,3,4,2]) # an,rfn,h,w,3
        que_imgs = color_map_backward(que_imgs.cpu().numpy()).transpose([0,2,3,1]) # qn,h,w,3
        imgs_out=[]
        for qi in range(que_imgs.shape[0]):
            que_img = que_imgs[qi] # h,w,3
            h, w, _ = que_img.shape
            gt_rot_img_gt, _ = transformation_crop(que_img,np.asarray([w/2,h/2],np.float32),1.0,-angles_gt[qi],h)
            rot_img_gt, _ = transformation_crop(que_img,np.asarray([w/2,h/2],np.float32),1.0,-angles_pr[qi,order_gt[qi,0]],h)
            rot_img_pr, _ = transformation_crop(que_img,np.asarray([w/2,h/2],np.float32),1.0,-angles_pr[qi,order_pr[qi,0]],h)
            gt_imgs = [que_img, gt_rot_img_gt]
            pr_imgs = [rot_img_gt, rot_img_pr]
            gt_imgs += [ref_imgs[2,k] for k in order_gt[qi,:5]]
            pr_imgs += [ref_imgs[2,k] for k in order_pr[qi,:5]]
            imgs_out.append(concat_images_list(concat_images_list(*gt_imgs),concat_images_list(*pr_imgs),vert=True))

        Path(f'{output_root}/{model_name}').mkdir(exist_ok=True, parents=True)
        imsave(f'{output_root}/{model_name}/{step}-{data_index}-region.jpg',concat_images_list(*imgs_out,vert=True))
        return outputs

class RefinerMetrics(Loss):
    default_cfg={
        "output_interval": 15,
        "scale_log_base": 2,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        # 'quaternion': quaternion, 'offset': offset, 'scale': scale
        quat_pr = data_pr['rotation'].cpu().numpy() # b,4
        offset_pr = data_pr['offset'].cpu().numpy() # b,2
        scale_pr = data_pr['scale'].cpu().numpy() # b,1

        quat_gt = data_gt['rotation'].cpu().numpy() # b,4
        offset_gt = data_gt['offset'].cpu().numpy()[:,:2] # b,2
        scale_gt = data_gt['scale'].cpu().numpy() # b,

        outputs = {}
        offset_err = np.linalg.norm(offset_pr - offset_gt,2,1) # b
        offset_acc_01 = (offset_err < 0.1).astype(np.float32)
        offset_acc_02 = (offset_err < 0.2).astype(np.float32)
        offset_acc_03 = (offset_err < 0.3).astype(np.float32)
        outputs.update({'off_acc_01': offset_acc_01, 'off_acc_02': offset_acc_02, 'off_acc_03': offset_acc_03,})

        rot_gt = [quat2mat(quat) for quat in quat_gt]
        rot_pr = [quat2mat(quat) for quat in quat_pr]
        rot_err = [mat2axangle(gt.T @ pr)[1] for gt, pr in zip(rot_gt, rot_pr)]
        rot_err = np.abs(np.rad2deg(rot_err))
        rot_acc_5 = (rot_err<5).astype(np.float32)
        rot_acc_10 = (rot_err<10).astype(np.float32)
        rot_acc_15 = (rot_err<15).astype(np.float32)
        outputs.update({'rot_acc_5': rot_acc_5, 'rot_acc_10': rot_acc_10, 'rot_acc_15': rot_acc_15,})

        scale_pr = self.cfg['scale_log_base'] ** scale_pr[...,0]
        scale_err = np.abs(np.log2(scale_pr/scale_gt))
        scale_acc_005 = (scale_err<0.05).astype(np.float32)
        scale_acc_003 = (scale_err<0.03).astype(np.float32)
        scale_acc_001 = (scale_err<0.01).astype(np.float32)
        outputs.update({'sc_acc_001': scale_acc_001, 'sc_acc_003': scale_acc_003, 'sc_acc_005': scale_acc_005})

        # estimate pose
        que_imgs_info = data_gt['que_imgs_info']
        poses_raw_gt = que_imgs_info['poses_raw'].cpu().numpy()
        Ks_raw = que_imgs_info['Ks_raw'].cpu().numpy()
        Ks_que = que_imgs_info['Ks'].cpu().numpy()
        Ks_in = que_imgs_info['Ks_in'].cpu().numpy()
        poses_rect = que_imgs_info['poses_rect'].cpu().numpy()

        poses_in = que_imgs_info['poses_in'].cpu().numpy()
        # poses_sim_in_to_que = que_imgs_info['poses_sim_in_to_que'].cpu().numpy()
        object_points = data_gt['object_points'].cpu().numpy()
        object_diameter = data_gt['object_diameter'].cpu().numpy()
        object_center = data_gt['object_center'].cpu().numpy()

        qn = object_center.shape[0]
        prj_errs, obj_errs, pose_errs, pose_pr_list = [], [], [], []
        for qi in range(qn):
            offset = np.concatenate([offset_pr[qi],np.zeros(1)])
            scale = scale_pr[qi]
            rotation = quat2mat(quat_pr[qi])
            center_in = pose_apply(poses_in[qi], object_center[qi])
            center_que = center_in + offset
            offset = center_que - (scale * rotation @ center_in)
            pose_sim_in_to_que = np.concatenate([scale * rotation, offset[:,None]],1)

            pose_in = poses_in[qi]
            pose_que = pose_sim_to_pose_rigid(pose_sim_in_to_que, pose_in, Ks_que[qi], Ks_in[qi], object_center[qi]) # obj to que
            pose_rect = poses_rect[qi]
            que_pose_pr = pose_compose(pose_que, pose_inverse(pose_rect)) # obj to raw
            pose_pr_list.append(que_pose_pr)
            que_pose_gt = poses_raw_gt[qi]

            prj_err, obj_err, pose_err = compute_pose_errors(object_points[qi],que_pose_pr,que_pose_gt,Ks_raw[qi])
            prj_errs.append(prj_err)
            obj_errs.append(obj_err)
            pose_errs.append(pose_err)

        prj_errs = np.stack(prj_errs, 0)
        obj_errs = np.stack(obj_errs, 0)
        pose_errs = np.stack(pose_errs, 0)
        add_01 = np.asarray(obj_errs<object_diameter*0.1,np.float32)
        prj_5 = np.asarray(prj_errs<5,np.float32)

        outputs.update({'prj_errs': prj_errs, 'obj_errs': obj_errs, 'R_errs': pose_errs[:,0], 't_errs': pose_errs[:,1], 'add_01': add_01, 'prj_5': prj_5})

        data_index=kwargs['data_index']
        if data_index % self.cfg['output_interval']!=0:
            return outputs
        model_name=kwargs['model_name']
        output_root = kwargs['output_root'] if 'output_root' in kwargs else 'data/vis'

        que_imgs = color_map_backward(data_gt['que_imgs_info']['imgs'].cpu().numpy()).transpose([0,2,3,1])
        ref_imgs = color_map_backward(data_gt['ref_imgs_info']['imgs'].cpu().numpy()).transpose([0,1,3,4,2])
        qn, h_, w_, _ = que_imgs.shape

        # visualize bbox
        que_img_raw = color_map_backward(que_imgs_info['imgs_raw'].cpu().numpy()).transpose([0,2,3,1])
        bbox_imgs = []
        for qi in range(min(qn,4)):
            # compute the initial pose
            object_bbox_3d = pts_range_to_bbox_pts(np.max(object_points[qi], 0), np.min(object_points[qi], 0))
            pose_pr = pose_pr_list[qi]
            pose_gt = poses_raw_gt[qi]
            pose_in = que_imgs_info['pose_in_raw'][qi].cpu().numpy()

            bbox_pts_pr, _ = project_points(object_bbox_3d,pose_pr,Ks_raw[qi])
            bbox_pts_gt, _ = project_points(object_bbox_3d,pose_gt,Ks_raw[qi])
            bbox_pts_in, _ = project_points(object_bbox_3d,pose_in,Ks_raw[qi])
            bbox_img = draw_bbox_3d(que_img_raw[qi], bbox_pts_gt)
            bbox_img = draw_bbox_3d(bbox_img, bbox_pts_in, (255,0,0))
            bbox_img = draw_bbox_3d(bbox_img, bbox_pts_pr, (0,0,255))
            bbox_img = concat_images_list(bbox_img, concat_images_list(que_imgs[qi],*(ref_imgs[qi]), vert=True))
            bbox_imgs.append(bbox_img)

        Path(f'{output_root}/{model_name}').mkdir(exist_ok=True, parents=True)
        imsave(f'{output_root}/{model_name}/{step}-{data_index}-bbox.jpg',concat_images_list(*bbox_imgs))
        return outputs


name2metrics={
    'vis_bbox_scale': VisualizeBBoxScale,
    'vis_sel': VisualizeSelector,
    'ref_metrics': RefinerMetrics
}

def selector_ang_acc(results):
    return np.mean(results['sel_acc_3'])+np.mean(results['sel_ang_5'])

def mean_iou(results):
    return np.mean(results['iou'])

def pose_add(results):
    return np.mean(results['add_01'])

name2key_metrics={
    'mean_iou': mean_iou,
    'sel_ang_acc': selector_ang_acc,
    'pose_add': pose_add,
}