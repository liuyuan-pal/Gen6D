import torch.nn.functional as F

from dataset.database import get_object_center, get_diameter, get_object_vert
from utils.base_utils import *
from utils.pose_utils import scale_rotation_difference_from_cameras, let_me_look_at, let_me_look_at_2d


def look_at_crop(img, K, pose, position, angle, scale, h, w):
    """rotate the image with "angle" and resize it with "scale", then crop the image on "position" with (h,w)"""
    # this function will return
    # 1) the resulted pose (pose_new) and intrinsic (K_new);
    # 2) pose_new = pose_compose(pose, pose_rect): "pose_rect" is the difference between the "pose_new" and the "pose"
    # 3) H is the homography that transform the "img" to "img_new"
    R_new, f_new = let_me_look_at_2d(position, K)
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
    R_new = R_z @ R_new
    f_new = f_new * scale
    K_new = np.asarray([[f_new, 0, w / 2], [0, f_new, h / 2], [0, 0, 1]], np.float32)

    H = K_new @ R_new @ np.linalg.inv(K)
    img_new = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    pose_rect = np.concatenate([R_new, np.zeros([3, 1])], 1).astype(np.float32)
    pose_new = pose_compose(pose, pose_rect)
    return img_new, K_new, pose_new, pose_rect, H

def compute_normalized_view_correlation(que_poses,ref_poses, center, th=True):
    """
    @param que_poses: [qn,3,4]
    @param ref_poses: [rfn,3,4]
    @param center:    [3]
    @param th:
    @return:
    """
    if th:
        que_cams = (que_poses[:,:,:3].permute(0,2,1) @ -que_poses[:,:,3:])[...,0] # qn,3
        ref_cams = (ref_poses[:,:,:3].permute(0,2,1) @ -ref_poses[:,:,3:])[...,0] # rfn,3
        que_diff = que_cams - center[None]
        ref_diff = ref_cams - center[None]
        que_diff = F.normalize(que_diff, dim=1)
        ref_diff = F.normalize(ref_diff, dim=1)
        corr = torch.sum(que_diff[:,None] * ref_diff[None,:], 2)
    else:
        que_cams = (que_poses[:,:,:3].transpose([0,2,1]) @ -que_poses[:,:,3:])[...,0] # qn,3
        ref_cams = (ref_poses[:,:,:3].transpose([0,2,1]) @ -ref_poses[:,:,3:])[...,0] # rfn,3
        # normalize to the same sphere
        que_cams = que_cams - center[None,:]
        ref_cams = ref_cams - center[None,:]
        que_cams = que_cams / np.linalg.norm(que_cams,2,1,keepdims=True)
        ref_cams = ref_cams / np.linalg.norm(ref_cams,2,1,keepdims=True)
        corr = np.sum(que_cams[:,None,:]*ref_cams[None,:,:], 2) # qn,rfn
    return corr

def normalize_reference_views(database, ref_ids, size, margin,
                              rectify_rot=True, input_pose=None, input_K=None,
                              add_rots=False, rots_list=None):
    object_center = get_object_center(database)
    object_diameter = get_diameter(database)

    ref_poses = np.asarray([database.get_pose(ref_id) for ref_id in ref_ids]) # rfn,3,3
    ref_Ks = np.asarray([database.get_K(ref_id) for ref_id in ref_ids]) # rfn,3,3
    ref_cens = np.asarray([project_points(object_center[None],pose, K)[0][0] for pose,K in zip(ref_poses, ref_Ks)]) # rfn,2
    ref_cams = np.stack([pose_inverse(pose)[:,3] for pose in ref_poses], 0) # rfn, 3

    # ensure that the output reference images have the same scale
    ref_dist = np.linalg.norm(ref_cams - object_center[None,], 2, 1) # rfn
    ref_focal_look = np.asarray([let_me_look_at(pose, K, object_center)[1] for pose, K in zip(ref_poses, ref_Ks)]) # rfn
    ref_focal_new = size * (1 - margin) / object_diameter * ref_dist
    ref_scales = ref_focal_new / ref_focal_look

    # ref_vert_angle will rotate the reference image to ensure the "up" direction approximate the Y- of the image
    if rectify_rot:
        if input_K is not None and input_pose is not None:
            # optionally, we may want to rotate the image with respect to a given pose so that they will be aligned.
            rfn = len(ref_poses)
            input_pose = np.repeat(input_pose[None, :], rfn, 0)
            input_K = np.repeat(input_K[None, :], rfn, 0)
            _, ref_vert_angle = scale_rotation_difference_from_cameras(ref_poses, input_pose, ref_Ks, input_K, object_center)  # rfn
        else:
            object_vert = get_object_vert(database)
            ref_vert_2d = np.asarray([(pose[:,:3] @ object_vert)[:2] for pose in ref_poses])
            mask = np.linalg.norm(ref_vert_2d,2,1)<1e-5
            ref_vert_2d[mask] += 1e-5 * np.sign(ref_vert_2d[mask]) # avoid 0 vector
            ref_vert_angle = -np.arctan2(ref_vert_2d[:,1],ref_vert_2d[:,0])-np.pi/2
    else:
        ref_vert_angle = np.zeros(len(ref_ids),np.float32)

    ref_imgs_new, ref_Ks_new, ref_poses_new, ref_Hs, ref_masks_new, ref_imgs_rots = [], [], [], [], [], []
    for k in range(len(ref_ids)):
        ref_img = database.get_image(ref_ids[k])
        if add_rots:
            ref_img_rot = np.stack([look_at_crop(ref_img, ref_Ks[k], ref_poses[k], ref_cens[k], ref_vert_angle[k]+rot, ref_scales[k], size, size)[0] for rot in rots_list],0)
            ref_imgs_rots.append(ref_img_rot)

        ref_img_new, ref_K_new, ref_pose_new, ref_pose_rect, ref_H = look_at_crop(
            ref_img, ref_Ks[k], ref_poses[k], ref_cens[k], ref_vert_angle[k], ref_scales[k], size, size)
        ref_imgs_new.append(ref_img_new)
        ref_Ks_new.append(ref_K_new)
        ref_poses_new.append(ref_pose_new)
        ref_Hs.append(ref_H)
        ref_mask = database.get_mask(ref_ids[k]).astype(np.float32)
        ref_masks_new.append(cv2.warpPerspective(ref_mask, ref_H, (size, size), flags=cv2.INTER_LINEAR))

    ref_imgs_new, ref_Ks_new, ref_poses_new, ref_Hs, ref_masks_new = \
        np.stack(ref_imgs_new, 0), np.stack(ref_Ks_new,0), np.stack(ref_poses_new,0), np.stack(ref_Hs,0), np.stack(ref_masks_new,0)

    if add_rots:
        ref_imgs_rots = np.stack(ref_imgs_rots,1)
        return ref_imgs_new, ref_masks_new, ref_Ks_new, ref_poses_new, ref_Hs, ref_imgs_rots
    return ref_imgs_new, ref_masks_new, ref_Ks_new, ref_poses_new, ref_Hs

def select_reference_img_ids_fps(database, ref_ids_all, ref_num, random_fps=False):
    object_center = get_object_center(database)
    # select ref ids
    poses = [database.get_pose(ref_id) for ref_id in ref_ids_all]
    cam_pts = np.asarray([pose_inverse(pose)[:, 3] - object_center for pose in poses])
    if random_fps:
        idxs = sample_fps_points(cam_pts, ref_num, False, index_model=True)
    else:
        idxs = sample_fps_points(cam_pts, ref_num + 1, True, index_model=True)

    ref_ids = np.asarray(ref_ids_all)[idxs]  # rfn
    return ref_ids

def select_reference_img_ids_refinement(ref_database, object_center, ref_ids, sel_pose, refine_ref_num=6, refine_even_ref_views=False, refine_even_num=128):
    ref_ids = np.asarray(ref_ids)
    ref_poses_all = np.asarray([ref_database.get_pose(ref_id) for ref_id in ref_ids])
    if refine_even_ref_views:
        # use fps to resample the reference images to make them distribute more evenly
        ref_cams_all = np.asarray([pose_inverse(pose)[:, 3] for pose in ref_poses_all])
        idx = sample_fps_points(ref_cams_all, refine_even_num + 1, True, index_model=True)
        ref_ids = ref_ids[idx]
        ref_poses_all = ref_poses_all[idx]

    corr = compute_normalized_view_correlation(sel_pose[None], ref_poses_all, object_center, False)
    ref_idxs = np.argsort(-corr[0])
    ref_idxs = ref_idxs[:refine_ref_num]
    ref_ids = ref_ids[ref_idxs]
    return ref_ids

# def normalize_reference_views(database: BaseDatabase, ref_ids_all, ref_num=32, size=128, margin=0.05):
#     ref_ids = select_reference_img_ids_fps(database, ref_ids_all, ref_num)
#     ref_imgs_new, ref_masks_new, ref_Ks_new, ref_poses_new, ref_Hs = construct_reference_views(database, ref_ids, size, margin)
#     return ref_imgs_new, ref_masks_new, ref_Ks_new, ref_poses_new, ref_Hs, ref_ids
