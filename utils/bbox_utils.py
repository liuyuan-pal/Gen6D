import torch
import numpy as np

def bboxes_lthw_squared(bboxes):
    """
    @param bboxes: b,4 in lthw
    @return:  b,4
    """
    bboxes_len = bboxes[:, 2:]
    bboxes_cen = bboxes[:, :2] + bboxes_len/2
    bboxes_max_len = torch.max(bboxes_len,1,keepdim=True)[0] # b,1
    bboxes_len = bboxes_max_len.repeat(1,2)
    bboxes_left_top = bboxes_cen - bboxes_len/2
    return torch.cat([bboxes_left_top,bboxes_len],1)

def bboxes_area(bboxes):
    return (bboxes[...,2]-bboxes[...,0])*(bboxes[...,3]-bboxes[...,1])

def bboxes_iou(bboxes0, bboxes1,th=True):
    """
    @param bboxes0: ...,4
    @param bboxes1: ...,4
    @return: ...
    """
    if th:
        x0 = torch.max(torch.stack([bboxes0[..., 0], bboxes1[..., 0]], -1), -1)[0]
        y0 = torch.max(torch.stack([bboxes0[..., 1], bboxes1[..., 1]], -1), -1)[0]
        x1 = torch.min(torch.stack([bboxes0[..., 2], bboxes1[..., 2]], -1), -1)[0]
        y1 = torch.min(torch.stack([bboxes0[..., 3], bboxes1[..., 3]], -1), -1)[0]
        inter = torch.clip(x1 - x0, min=0) * torch.clip(y1 - y0, min=0)
    else:
        x0 = np.max(np.stack([bboxes0[..., 0], bboxes1[..., 0]], -1), -1)[0]
        y0 = np.max(np.stack([bboxes0[..., 1], bboxes1[..., 1]], -1), -1)[0]
        x1 = np.min(np.stack([bboxes0[..., 2], bboxes1[..., 2]], -1), -1)[0]
        y1 = np.min(np.stack([bboxes0[..., 3], bboxes1[..., 3]], -1), -1)[0]
        inter = np.clip(x1 - x0, a_min=0, a_max=999999) * np.clip(y1 - y0, a_min=0, a_max=999999)
    union = bboxes_area(bboxes0) + bboxes_area(bboxes1) - inter
    iou = inter / union
    return iou

def lthw_to_ltrb(bboxes,th=True):
    if th:
        return torch.cat([bboxes[...,:2],bboxes[...,:2]+bboxes[...,2:]],-1)
    else:
        return np.concatenate([bboxes[..., :2], bboxes[..., :2] + bboxes[..., 2:]], -1)

def cl_to_ltrb(bboxes_cl):
    bboxes_cen = bboxes_cl[...,:2]
    bboxes_len = bboxes_cl[...,2:]
    return torch.cat([bboxes_cen-bboxes_len/2,bboxes_cen+bboxes_len/2],-1)

def ltrb_to_cl(bboxes_ltrb):
    bboxes_cen = (bboxes_ltrb[...,:2]+bboxes_ltrb[...,2:])/2
    bboxes_len = bboxes_ltrb[..., 2:]-bboxes_ltrb[...,:2]
    return torch.cat([bboxes_cen,bboxes_len],-1)

def ltrb_to_lthw(bboxes,th=True):
    if th:
        raise NotImplementedError
    else:
        lt = bboxes[...,:2]
        hw = bboxes[...,2:] - lt
        return np.concatenate([lt,hw],-1)

def cl_to_lthw(bboxes_cl,th=True):
    if th:
        lt = bboxes_cl[..., :2] - bboxes_cl[..., 2:] / 2
        return torch.cat([lt, bboxes_cl[..., 2:]], -1)
    else:
        lt = bboxes_cl[..., :2] - bboxes_cl[..., 2:] / 2
        return np.concatenate([lt,bboxes_cl[...,2:]],-1)

def parse_bbox_from_scale_offset(que_select_id, scale_pr, select_offset, pool_ratio, ref_shape):
    """

    @param que_select_id:  [2] x,y
    @param scale_pr:       [hq,wq]
    @param select_offset:  [2,hq,wq]
    @param pool_ratio:     int
    @param ref_shape:      [2] h,w
    @return:
    """
    hr, wr = ref_shape
    select_x, select_y = que_select_id
    scale_pr = scale_pr
    offset_pr = select_offset
    scale_pr = scale_pr[select_y,select_x]
    scale_pr = 2**scale_pr
    pool_ratio = pool_ratio
    offset_x, offset_y = offset_pr[:,select_y,select_x]
    center_x, center_y = select_x+offset_x, select_y+offset_y
    center_x = (center_x + 0.5) * pool_ratio - 0.5
    center_y = (center_y + 0.5) * pool_ratio - 0.5
    h_pr, w_pr = hr * scale_pr, wr * scale_pr
    bbox_pr = np.asarray([center_x - w_pr/2, center_y-h_pr/2, w_pr, h_pr])
    return bbox_pr