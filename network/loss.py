import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.operator import generate_coords, pose_apply_th
from pytorch3d.transforms import quaternion_apply


class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys=keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class DetectionSoftmaxLoss(Loss):
    default_cfg={
        'score_diff_thresh': 1.5,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__(['loss_cls','acc_loc'])
        self.loss_op = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, data_pr, data_gt, step, **kwargs):
        center = data_gt['que_imgs_info']['cens'] # qn,2
        pool_ratio = data_pr['pool_ratio']
        center = (center + 0.5) / pool_ratio - 0.5 # qn,2

        # generate label
        scores = data_pr['scores']
        qn,_, h, w = scores.shape
        coords = generate_coords(h, w, scores.device) # h,w,2
        coords = coords.unsqueeze(0).repeat(qn,1,1,1).permute(0,3,1,2) # qn,2,h,w
        center = center[:,:,None,None] # qn,2,h,w
        labels = (torch.norm(coords-center,dim=1)<self.cfg['score_diff_thresh']).float() # qn,h,w
        scores, labels = scores.flatten(1), labels.flatten(1) # [qn,h*w] [qn,h*w]

        loss = self.loss_op(scores, labels)
        loss_pos = torch.sum(loss * labels, 1)/ (torch.sum(labels,1)+1e-3)
        loss_neg = torch.sum(loss * (1-labels), 1)/ (torch.sum(1-labels,1)+1e-3)
        loss = (loss_pos+loss_neg) / 2.0

        return {'loss_cls': loss}

class DetectionOffsetAndScaleLoss(Loss):
    default_cfg={
        'diff_thresh': 1.5,
        'scale_ratio': 1.0,
        'use_offset_loss': True,
        'use_angle_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super(DetectionOffsetAndScaleLoss, self).__init__(['loss_scale','loss_offset'])

    # @staticmethod
    # def _check_offset(offset_diff,name):
    #     from utils.base_utils import color_map_backward
    #     from skimage.io import imsave
    #     offset_diff = offset_diff.detach().cpu().numpy()
    #     offset_diff = 1/(1+offset_diff)
    #     imsave(name, color_map_backward(offset_diff[0]))
    #     print('check mode is on !!!')

    def _loss(self, offset_pr, scale_pr, center, scale_gt):
        """

        @param offset_pr: [qn,2,h,w]
        @param scale_pr:  [qn,1,h,w]
        @param center:    [qn,2]
        @param scale_gt:  [qn]
        @return:
        """
        qn, _, h, w = offset_pr.shape
        coords = generate_coords(h, w, offset_pr.device) # h,w,2
        coords = coords.unsqueeze(0).repeat(qn,1,1,1).permute(0,3,1,2) # qn,2,h,w
        center = center[:,:,None,None].repeat(1,1,h,w) # qn,2,h,w
        diff = center - coords # qn,2,h,w
        mask = torch.norm(diff,2,1)<self.cfg['diff_thresh'] # qn, h, w
        mask = mask.float()

        scale_gt = torch.log2(scale_gt)
        scale_diff = (scale_pr - scale_gt[:, None, None, None]) ** 2
        loss_scale = torch.sum(scale_diff.flatten(1)*mask.flatten(1),1) / (torch.sum(mask.flatten(1),1)+1e-4)
        if self.cfg['use_offset_loss']:
            offset_diff = torch.sum((offset_pr - diff) ** 2, 1) # qn, h, w
            loss_offset = torch.sum(offset_diff.flatten(1)*mask.flatten(1),1) / (torch.sum(mask.flatten(1),1)+1e-4)
        else:
            loss_offset = torch.zeros_like(loss_scale)
        return loss_offset, loss_scale

    def __call__(self, data_pr, data_gt, step, **kwargs):
        center = data_gt['que_imgs_info']['cens']
        pool_ratio = data_pr['pool_ratio']
        center = (center + 0.5) / pool_ratio - 0.5 # qn,2

        loss_offset, loss_scale = self._loss(data_pr['select_pr_offset'], data_pr['select_pr_scale'], center, data_gt['scale_diff'])  # qn
        loss_scale = self.cfg['scale_ratio'] * loss_scale
        return {'loss_scale': loss_scale, 'loss_offset': loss_offset}


class SelectionLoss(Loss):
    default_cfg={
        "normalize_gt_score": True,
    }
    def __init__(self,cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([])
        self.bce_loss=nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, data_pr, data_gt, step, **kwargs):
        logits_pr = data_pr['ref_vp_logits'] # qn,rfn
        scores_gt = data_gt['ref_vp_scores'] # qn,rfn

        # scale scores_gt to [0,1]
        # todo: maybe we can scale scores to softmax, normalize scores_gt to sum = 1.0.
        if self.cfg['normalize_gt_score']:
            scores_gt_min = torch.min(scores_gt,1,keepdim=True)[0]
            scores_gt_max = torch.max(scores_gt,1,keepdim=True)[0]
            scores_gt = (scores_gt - scores_gt_min) / torch.clamp(scores_gt_max - scores_gt_min, min=1e-4)
        else:
            scores_gt = (scores_gt + 1) / 2
        loss_score = self.bce_loss(logits_pr, scores_gt)
        loss_score = torch.mean(loss_score,1)

        # angle loss
        angles_pr = data_pr['angles_pr'] # qn, rfn
        angles_gt = data_gt['angles_r2q'] # qn,
        ref_ids_gt = data_gt['gt_ref_ids'] # qn
        qn, rfn = angles_pr.shape
        angles_pr = angles_pr[torch.arange(qn), ref_ids_gt]
        angles_gt = angles_gt*2/np.pi # scale [-90,90] to [-1,1]
        loss_angle = (angles_pr - angles_gt)**2
        return {'loss_score': loss_score, 'loss_angle': loss_angle}

class RefinerLoss(Loss):
    default_cfg={
        "scale_log_base": 2,
        "loss_space": 'sim',
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([])

    @staticmethod
    def apply_rigid_transformation(grids, center, scale, offset, quaternion):
        """
        @param grids:       [qn,pn,3]
        @param center:      [qn,1,3]
        @param scale:       [qn,1]
        @param offset:      [qn,2]
        @param quaternion:  [qn,4]
        @return:
        """
        pn = grids.shape[1]
        grids_ = quaternion_apply(quaternion[:, None].repeat(1, pn, 1), grids - center) # rotate
        center[:, :, :2] += offset[:, None, :2] # 2D offset
        center[:, :, 2:] *= scale[:, None, :] # scale
        grids_ = grids_ + center
        return grids_

    def __call__(self, data_pr, data_gt, step, **kwargs):
        quaternion_pr = data_pr['rotation'] # qn,4
        offset_pr = data_pr['offset'] # qn,2
        scale_pr = data_pr['scale'] # qn,1

        center = data_gt['object_center'] # qn,3
        poses_in = data_gt['que_imgs_info']['poses_in'] # qn,3,4
        center_in = pose_apply_th(poses_in, center[:,None,:]) # qn,1,3

        grids = data_pr['grids'] # qn,pn,3
        pn = grids.shape[1]
        if self.cfg['loss_space'] == 'sim':
            grids_pr = (self.cfg['scale_log_base'] ** scale_pr[:,None]) * quaternion_apply(quaternion_pr[:,None].repeat(1,pn,1), grids - center_in) + center_in
            grids_pr[...,:2] = grids_pr[...,:2] + offset_pr[:,None,:2]
            grids_gt = pose_apply_th(data_gt['que_imgs_info']['poses_sim_in_to_que'], grids)
        elif self.cfg['loss_space'] == 'raw':
            scale_gt, offset_gt, quaternion_gt = data_gt['scale'].unsqueeze(1), data_gt['offset'], data_gt['rotation']
            grids_gt = self.apply_rigid_transformation(grids, center_in, scale_gt, offset_gt, quaternion_gt)
            scale_pr = self.cfg['scale_log_base'] ** scale_pr
            grids_pr = self.apply_rigid_transformation(grids, center_in, scale_pr, offset_pr, quaternion_pr)
        else:
            raise NotImplementedError

        loss = torch.mean(torch.sum((grids_gt - grids_pr)**2,-1), 1)
        return {'loss_pose': loss}

name2loss={
    'detection_softmax': DetectionSoftmaxLoss,
    'detection_offset_scale': DetectionOffsetAndScaleLoss,
    'selection_loss': SelectionLoss,
    'refiner_loss': RefinerLoss,
}