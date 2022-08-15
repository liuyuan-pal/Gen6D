import argparse
from copy import copy
from pathlib import Path

import numpy as np
from skimage.io import imsave

from tqdm import tqdm

from dataset.database import parse_database_name, get_database_split, get_ref_point_cloud, get_diameter, get_object_center
from estimator import name2estimator
from utils.base_utils import load_cfg, save_pickle, read_pickle, project_points, transformation_crop
from utils.database_utils import compute_normalized_view_correlation
from utils.draw_utils import draw_bbox, concat_images_list, draw_bbox_3d, pts_range_to_bbox_pts
from utils.pose_utils import compute_metrics_impl, scale_rotation_difference_from_cameras


def get_gt_info(que_pose, que_K, render_poses, render_Ks, object_center):
    gt_corr = compute_normalized_view_correlation(que_pose[None], render_poses, object_center, False)
    gt_ref_idx = np.argmax(gt_corr[0])
    gt_scale_r2q, gt_angle_r2q = scale_rotation_difference_from_cameras(
        render_poses[gt_ref_idx][None], que_pose[None], render_Ks[gt_ref_idx][None], que_K[None], object_center)
    gt_scale_r2q, gt_angle_r2q = gt_scale_r2q[0], gt_angle_r2q[0]
    gt_position = project_points(object_center[None], que_pose, que_K)[0][0]
    size = 128
    gt_bbox = np.concatenate([gt_position - size / 2 * gt_scale_r2q, np.full(2, size) * gt_scale_r2q])
    return gt_position, gt_scale_r2q, gt_angle_r2q, gt_ref_idx, gt_bbox, gt_corr[0]

def visualize_intermediate_results(img, K, inter_results, ref_info, object_bbox_3d, object_center=None, pose_gt=None):
    ref_imgs = ref_info['ref_imgs']  # an,rfn,h,w,3
    if pose_gt is not None:
        gt_position, gt_scale_r2q, gt_angle_r2q, gt_ref_idx, gt_bbox, gt_scores = \
            get_gt_info(pose_gt, K, ref_info['poses'], ref_info['Ks'], object_center)

    output_imgs = []
    if 'det_scale_r2q' in inter_results and 'sel_angle_r2q' in inter_results:
        # visualize detection
        det_scale_r2q = inter_results['det_scale_r2q']
        det_position = inter_results['det_position']
        det_que_img = inter_results['det_que_img']
        size = det_que_img.shape[0]
        pr_bbox = np.concatenate([det_position - size / 2 * det_scale_r2q, np.full(2, size) * det_scale_r2q])
        bbox_img = img
        if pose_gt is not None: bbox_img = draw_bbox(bbox_img, gt_bbox, color=(0, 255, 0))
        bbox_img = draw_bbox(bbox_img, pr_bbox, color=(0, 0, 255))
        output_imgs.append(bbox_img)

        # visualize selection
        sel_angle_r2q = inter_results['sel_angle_r2q']  #
        sel_scores = inter_results['sel_scores']  #
        h, w, _ = det_que_img.shape
        sel_img_rot, _ = transformation_crop(det_que_img, np.asarray([w / 2, h / 2], np.float32), 1.0, -sel_angle_r2q, h)
        an = ref_imgs.shape[0]
        sel_img = concat_images_list(det_que_img, sel_img_rot, *[ref_imgs[an // 2, score_idx] for score_idx in np.argsort(-sel_scores)[:5]], vert=True)
        if pose_gt is not None:
            sel_img_rot_gt, _ = transformation_crop(det_que_img, np.asarray([w/2, h/2], np.float32), 1.0, -gt_angle_r2q, h)
            sel_img_gt = concat_images_list(det_que_img, sel_img_rot_gt, *[ref_imgs[an // 2, score_idx] for score_idx in np.argsort(-gt_scores)[:5]], vert=True)
            sel_img = concat_images_list(sel_img, sel_img_gt)
        output_imgs.append(sel_img)

    # visualize refinements
    refine_poses = inter_results['refine_poses']
    refine_imgs = []
    for k in range(1,len(refine_poses)):
        pose_in, pose_out = refine_poses[k-1], refine_poses[k]
        bbox_pts_in, _ = project_points(object_bbox_3d, pose_in, K)
        bbox_pts_out, _ = project_points(object_bbox_3d, pose_out, K)
        bbox_img = draw_bbox_3d(img, bbox_pts_in, (255, 0, 0))
        if pose_gt is not None:
            bbox_pts_gt, _ = project_points(object_bbox_3d, pose_gt, K)
            bbox_img = draw_bbox_3d(bbox_img, bbox_pts_gt, (0, 255, 0))
        bbox_img = draw_bbox_3d(bbox_img, bbox_pts_out, (0, 0, 255))
        refine_imgs.append(bbox_img)
    output_imgs.append(concat_images_list(*refine_imgs))
    return concat_images_list(*output_imgs)

def visualize_final_poses(img, K, object_bbox_3d, pose_pr, pose_gt=None):
    bbox_pts_pr, _ = project_points(object_bbox_3d, pose_pr, K)
    bbox_img = img
    if pose_gt is not None:
        bbox_pts_gt, _ = project_points(object_bbox_3d, pose_gt, K)
        bbox_img = draw_bbox_3d(bbox_img, bbox_pts_gt)
    bbox_img = draw_bbox_3d(bbox_img, bbox_pts_pr, (0, 0, 255))
    return bbox_img


def main(args):
    # estimator
    cfg = load_cfg(args.cfg)
    object_name = args.object_name
    if object_name.startswith('linemod'):
        ref_database_name = que_database_name = object_name
        que_split = 'linemod_test'
    elif object_name.startswith('genmop'):
        ref_database_name = object_name+'-ref'
        que_database_name = object_name+'-test'
        que_split = 'all'
    else:
        raise NotImplementedError

    ref_database = parse_database_name(ref_database_name)
    estimator = name2estimator[cfg['type']](cfg)
    ref_split = que_split if args.split_type is None else args.split_type
    estimator.build(ref_database, split_type=ref_split)

    que_database = parse_database_name(que_database_name)
    _, que_ids = get_database_split(que_database, que_split)

    object_pts = get_ref_point_cloud(ref_database)
    object_center = get_object_center(ref_database)
    object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

    est_name = estimator.cfg["name"] # + f'-{args.render_pose_name}'
    est_name = est_name + args.split_type if args.split_type is not None else est_name
    Path(f'data/eval/poses/{object_name}').mkdir(exist_ok=True,parents=True)
    Path(f'data/vis_inter/{est_name}/{object_name}').mkdir(exist_ok=True,parents=True)
    Path(f'data/vis_final/{est_name}/{object_name}').mkdir(exist_ok=True,parents=True)
    if not args.eval_only:
        pose_pr_list = []

        for que_id in tqdm(que_ids):
            # estimate pose
            img = que_database.get_image(que_id)
            K = que_database.get_K(que_id)
            pose_pr, inter_results = estimator.predict(img, K)
            pose_pr_list.append(pose_pr)

            pose_gt = que_database.get_pose(que_id)
            inter_img = visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d, object_center, pose_gt)
            imsave(f'data/vis_inter/{est_name}/{object_name}/{que_id}-inter.jpg', inter_img)
            final_img = visualize_final_poses(img, K, object_bbox_3d, pose_pr, pose_gt)
            imsave(f'data/vis_final/{est_name}/{object_name}/{que_id}-bbox3d.jpg', final_img)

        save_pickle(pose_pr_list, f'data/eval/poses/{object_name}/{est_name}.pkl')
    else:
        pose_pr_list = read_pickle(f'data/eval/poses/{object_name}/{est_name}.pkl')

    # evaluation metrics
    pose_gt_list = [que_database.get_pose(que_id) for que_id in que_ids]
    que_Ks = [que_database.get_K(que_id) for que_id in que_ids]
    object_diameter = get_diameter(que_database)

    def get_eval_msg(pose_in_list,msg_in,scale=1.0):
        msg_in = copy(msg_in)
        results = compute_metrics_impl(object_pts, object_diameter, pose_gt_list, pose_in_list, que_Ks, scale, symmetric=args.symmetric)
        for k, v in results.items(): msg_in+=f'{k} {v:.4f} '
        return msg_in + '\n'

    msg_pr = f'{object_name:10} {est_name:20} '
    msg_pr=get_eval_msg(pose_pr_list, msg_pr)
    print(msg_pr)
    with open('data/performance.log','a') as f: f.write(msg_pr)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--object_name', type=str, default='warrior')
    parser.add_argument('--eval_only', action='store_true', dest='eval_only', default=False)
    parser.add_argument('--symmetric', action='store_true', dest='symmetric', default=False)
    parser.add_argument('--split_type', type=str, default=None)
    args = parser.parse_args()
    main(args)


