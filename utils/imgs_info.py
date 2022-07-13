import torch
import numpy as np

from utils.base_utils import color_map_forward


def imgs_info_to_torch(imgs_info):
    for k, v in imgs_info.items():
        if isinstance(v,np.ndarray):
            imgs_info[k] = torch.from_numpy(v)
    return imgs_info

def build_imgs_info(database, ref_ids, has_mask=True):
    ref_Ks = np.asarray([database.get_K(ref_id) for ref_id in ref_ids], dtype=np.float32)

    ref_imgs = [database.get_image(ref_id) for ref_id in ref_ids]
    if has_mask: ref_masks =  [database.get_mask(ref_id) for ref_id in ref_ids]
    else: ref_masks = None

    ref_imgs = (np.stack(ref_imgs, 0)).transpose([0, 3, 1, 2])
    ref_imgs = color_map_forward(ref_imgs)
    if has_mask: ref_masks = np.stack(ref_masks, 0)[:, None, :, :]
    ref_poses = np.asarray([database.get_pose(ref_id) for ref_id in ref_ids], dtype=np.float32)

    ref_imgs_info = {'imgs': ref_imgs, 'poses': ref_poses, 'Ks': ref_Ks}
    if has_mask: ref_imgs_info['masks'] = ref_masks
    return ref_imgs_info
