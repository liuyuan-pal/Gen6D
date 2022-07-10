import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

from network.attention import AttentionBlock
from network.pretrain_models import VGGBNPretrain
from utils.base_utils import color_map_forward


class ViewpointSelector(nn.Module):
    default_cfg = {
        'selector_angle_num': 5,
    }
    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__()
        self.backbone = VGGBNPretrain([0,1,2])
        for para in self.backbone.parameters():
            para.requires_grad = False
        self.img_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.ref_feats_cache = None
        self.ref_pose_embed = None # rfn,f
        self.ref_angle_embed = None # an,f

        corr_conv0 = nn.Sequential(
            nn.InstanceNorm3d(512),
            nn.Conv3d(512,64,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64,64,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(64),
            nn.MaxPool3d((1,2,2),(1,2,2)),

            nn.Conv3d(64,128,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128,128,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(128),
            nn.MaxPool3d((1,2,2),(1,2,2)),

            nn.Conv3d(128,256,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256,256,(1,3,3),padding=(0,1,1)),
        )
        corr_conv1 = nn.Sequential(
            nn.InstanceNorm3d(512),
            nn.Conv3d(512,128,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128,128,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(128),
            nn.MaxPool3d((1,2,2),(1,2,2)),

            nn.Conv3d(128,256,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256,256,(1,3,3),padding=(0,1,1)),
        )
        corr_conv2 = nn.Sequential(
            nn.InstanceNorm3d(512),
            nn.Conv3d(512,256,(1,3,3),padding=(0,1,1)),
            nn.InstanceNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256,256,(1,3,3),padding=(0,1,1)),
        )
        self.corr_conv_list = nn.ModuleList([corr_conv0,corr_conv1,corr_conv2])

        self.corr_feats_conv = nn.Sequential(
            nn.Conv3d(256*3,512,1,1),
            nn.InstanceNorm3d(512),
            nn.ReLU(True),
            nn.Conv3d(512,512,1,1),
            nn.AvgPool3d((1,4,4))
        )
        self.vp_norm=nn.InstanceNorm2d(3)
        self.score_process = nn.Sequential(
            nn.Conv2d(3+512, 512, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, 1),
        )

        self.atts = [AttentionBlock(512, 512, 512, 8, skip_connect=False) for _ in range(2)]
        self.mlps = [nn.Sequential(nn.Conv1d(512*2,512,1,1),nn.InstanceNorm1d(512),nn.ReLU(True),
                                   nn.Conv1d(512,512,1,1),nn.InstanceNorm1d(512),nn.ReLU(True)) for _ in range(2)]

        self.mlps = nn.ModuleList(self.mlps)
        self.atts = nn.ModuleList(self.atts)
        self.score_predict = nn.Sequential(
            nn.Conv1d(512, 512, 1, 1),
            nn.ReLU(True),
            nn.Conv1d(512, 1, 1, 1),
        )

        an = self.cfg['selector_angle_num']
        self.angle_predict = nn.Sequential(
            nn.Conv1d((3+512) * an, 512, 1, 1),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 1, 1),
            nn.ReLU(True),
            nn.Conv1d(512, 1, 1, 1)
        )
        self.view_point_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
        )

    def get_feats(self, imgs):
        self.backbone.eval()
        imgs = self.img_norm(imgs)
        with torch.no_grad():
            feats_list = self.backbone(imgs)
            feats_list = [F.normalize(feats, dim=1) for feats in feats_list]
        return feats_list

    def extract_ref_feats(self, ref_imgs, ref_poses, object_center, object_vert, is_train=False):
        # get features
        an, rfn, _, h, w = ref_imgs.shape
        ref_feats = self.get_feats(ref_imgs.reshape(an * rfn, 3, h, w))

        ref_feats_out = []
        for feats in ref_feats:
            _, f, h, w = feats.shape
            ref_feats_out.append(feats.reshape(an, rfn, f, h, w))
        self.ref_feats_cache = ref_feats_out

        # get pose embedding
        ref_cam_pts = -ref_poses[:,:3,:3].permute(0,2,1) @ ref_poses[:,:3,3:] # rfn,3,3 @ rfn,3,1
        ref_cam_pts = ref_cam_pts[...,0] - object_center[None] # rfn,3
        if is_train:
            object_forward = ref_cam_pts[np.random.randint(0,ref_cam_pts.shape[0])]
        else:
            object_forward = ref_cam_pts[0]
        # get rotation
        y = torch.cross(object_vert, object_forward)
        x = torch.cross(y, object_vert)
        object_vert = F.normalize(object_vert,dim=0)
        x = F.normalize(x,dim=0)
        y = F.normalize(y,dim=0)
        R = torch.stack([x, y, object_vert], 0)
        ref_cam_pts = ref_cam_pts @ R.T # rfn,3 @ 3,3
        ref_cam_pts = F.normalize(ref_cam_pts,dim=1) # rfn, 3 --> normalized viewpoints here
        self.ref_pose_embed = self.view_point_encoder(ref_cam_pts) # rfn,512

    def load_ref_imgs(self, ref_imgs, ref_poses, object_center, object_vert):
        """
        @param ref_imgs: [an,rfn,h,w,3]
        @param ref_poses: [rfn,3,4]
        @param object_center: [3]
        @param object_vert: [3]
        @return:
        """
        an,rfn,h,w,_=ref_imgs.shape
        ref_imgs = torch.from_numpy(color_map_forward(ref_imgs).transpose([0, 1, 4, 2, 3])).cuda()  # an,rfn,3,h,w
        ref_poses, object_center, object_vert = torch.from_numpy(ref_poses.astype(np.float32)).cuda(), \
                                                torch.from_numpy(object_center.astype(np.float32)).cuda(), \
                                                torch.from_numpy(object_vert.astype(np.float32)).cuda()
        self.extract_ref_feats(ref_imgs, ref_poses, object_center, object_vert)

    def select_que_imgs(self, que_imgs):
        """
        @param que_imgs: [qn,h,w,3]
        @return:
        """
        que_imgs = torch.from_numpy(color_map_forward(que_imgs).transpose([0, 3, 1, 2])).cuda()  # qn,3,h,w
        logits, angles = self.compute_view_point_feats(que_imgs) # qn,rfn
        ref_idx = torch.argmax(logits,1) # qn,
        angles = angles[torch.arange(ref_idx.shape[0]), ref_idx] # qn,
        # qn, qn, [qn,rfn]
        return {'ref_idx': ref_idx.cpu().numpy(), 'angles': angles.cpu().numpy(), 'scores': logits.cpu().numpy()}

    def compute_view_point_feats(self, que_imgs):
        que_feats_list = self.get_feats(que_imgs) # qn,f,h,w
        ref_feats_list = self.ref_feats_cache # an,rfn,f,h,w

        vps_feats, corr_feats = [], []
        for ref_feats, que_feats, corr_conv in zip(ref_feats_list, que_feats_list, self.corr_conv_list):
            ref_feats = ref_feats.permute(1,0,2,3,4) # rfn,an,f,h,w
            feats_corr = que_feats[:, None,None] * ref_feats[None] # qn,rfn,an,f,h,w
            qn, rfn, an, f, h, w = feats_corr.shape
            feats_corr = feats_corr.permute(0,3,1,2,4,5).reshape(qn,f,rfn*an,h,w)
            feats_corr_ = corr_conv(feats_corr)
            _, f_, _, h_, w_ = feats_corr_.shape
            corr_feats.append(feats_corr_.reshape(qn,f_, rfn, an, h_, w_)) # qn,f_,rfn,an,h_,w_

            # vps score feats
            score_maps = torch.sum(feats_corr, 1) # qn,rfn*an,h,w
            score_maps_ = score_maps/(torch.max(score_maps.flatten(2),2)[0][...,None,None]) # qn,rfn*an,h,w
            score_vps = torch.sum(score_maps.flatten(2)*score_maps_.flatten(2),2) # qn,rfn*an
            vps_feats.append(score_vps.reshape(qn,rfn,an))

        corr_feats = torch.cat(corr_feats, 1)  # qn,f_*3,rfn,an,h_,w_
        qn, f, rfn, an, h, w = corr_feats.shape
        corr_feats = self.corr_feats_conv(corr_feats.reshape(qn, f, rfn*an, h, w))[...,0,0] # qn,f,rfn,an
        corr_feats = corr_feats.reshape(qn,corr_feats.shape[1],rfn,an)
        vps_feats = self.vp_norm(torch.stack(vps_feats, 1),) # qn,3,rfn,an
        feats = torch.cat([corr_feats, vps_feats],1) # qn,f+3,rfn,an

        scores_feats = torch.max(self.score_process(feats),3)[0] # qn,512,rfn
        scores_feats = scores_feats + self.ref_pose_embed.T.unsqueeze(0) # qn,512,rfn

        for att, mlp in zip(self.atts, self.mlps):
            msg = att(scores_feats, scores_feats) #
            scores_feats = mlp(torch.cat([scores_feats, msg], 1)) + scores_feats
        logits = self.score_predict(scores_feats)[:,0,:] # qn,rfn

        qn, f, rfn, an = feats.shape
        feats = feats.permute(0,1,3,2).reshape(qn,f*an,rfn)
        angles = self.angle_predict(feats)[:,0,:] # qn,rfn
        return logits, angles

    def forward(self, data):
        ref_imgs = data['ref_imgs'] # an,rfn,3,h,w
        ref_poses = data['ref_imgs_info']['poses']
        object_center = data['object_center']
        object_vert = data['object_vert']
        que_imgs = data['que_imgs_info']['imgs'] # qn,3,h,w
        is_train = 'eval' not in data
        self.extract_ref_feats(ref_imgs, ref_poses, object_center, object_vert, is_train)
        logits, angles = self.compute_view_point_feats(que_imgs)
        return {'ref_vp_logits': logits, 'angles_pr': angles}
