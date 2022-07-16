import argparse
import subprocess
from pathlib import Path

import numpy as np
from skimage.io import imsave, imread
from tqdm import tqdm

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from eval import visualize_intermediate_results
from prepare import video2image
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp


def weighted_pts(pts_list, weight_num=10, std_inv=10):
    weights=np.exp(-(np.arange(weight_num)/std_inv)**2)[::-1] # wn
    pose_num=len(pts_list)
    if pose_num<weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:,None,None],0)/np.sum(weights)
    return pts

def main(args):
    cfg = load_cfg(args.cfg)
    ref_database = parse_database_name(args.database)
    estimator = name2estimator[cfg['type']](cfg)
    estimator.build(ref_database, split_type='all')

    object_pts = get_ref_point_cloud(ref_database)
    object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    (output_dir / 'images_raw').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_out').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_inter').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_out_smooth').mkdir(exist_ok=True, parents=True)

    que_num = video2image(args.video, output_dir/'images_raw', 1, args.resolution, args.transpose)

    pose_init = None
    hist_pts = []
    for que_id in tqdm(range(que_num)):
        img = imread(str(output_dir/'images_raw'/f'frame{que_id}.jpg'))
        # generate a pseudo K
        h, w, _ = img.shape
        f=np.sqrt(h**2+w**2)
        K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)

        if pose_init is not None:
            estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization
        pose_pr, inter_results = estimator.predict(img, K, pose_init=pose_init)
        pose_init = pose_pr

        pts, _ = project_points(object_bbox_3d, pose_pr, K)
        bbox_img = draw_bbox_3d(img, pts, (0,0,255))
        imsave(f'{str(output_dir)}/images_out/{que_id}-bbox.jpg', bbox_img)
        np.save(f'{str(output_dir)}/images_out/{que_id}-pose.npy', pose_pr)
        imsave(f'{str(output_dir)}/images_inter/{que_id}.jpg', visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d))

        hist_pts.append(pts)
        pts_ = weighted_pts(hist_pts, weight_num=args.num, std_inv=args.std)
        pose_ = pnp(object_bbox_3d, pts_, K)
        pts__, _ = project_points(object_bbox_3d, pose_, K)
        bbox_img_ = draw_bbox_3d(img, pts__, (0,0,255))
        imsave(f'{str(output_dir)}/images_out_smooth/{que_id}-bbox.jpg', bbox_img_)


    cmd=[args.ffmpeg, '-y', '-framerate','30', '-r', '30',
         '-i', f'{output_dir}/images_out_smooth/%d-bbox.jpg',
         '-c:v', 'libx264','-pix_fmt','yuv420p', f'{output_dir}/video.mp4']
    subprocess.run(cmd)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain.yaml')
    parser.add_argument('--database', type=str, default="custom/mouse")
    parser.add_argument('--output', type=str, default="data/custom/mouse/test")

    # input video process
    parser.add_argument('--video', type=str, default="data/custom/video/mouse-test.mp4")
    parser.add_argument('--resolution', type=int, default=960)
    parser.add_argument('--transpose', action='store_true', dest='transpose', default=False)

    # smooth poses
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--std', type=float, default=2.5)

    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
    args = parser.parse_args()
    main(args)