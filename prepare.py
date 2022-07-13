import argparse
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from colmap_script import build_colmap_model_no_pose
from dataset.database import parse_database_name, get_database_split
from estimator import Gen6DEstimator
from network import name2network
from utils.base_utils import load_cfg, save_pickle


def video2image(input_video, output_dir, interval=30, image_size = 640, transpose=False):
    print(f'split video {input_video} into images ...')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % interval==0:
            h, w = image.shape[:2]
            ratio = image_size/max(h,w)
            ht, wt = int(ratio*h), int(ratio*w)
            image = cv2.resize(image,(wt,ht),interpolation=cv2.INTER_LINEAR)
            if transpose:
                image = cv2.transpose(image)
                image = cv2.flip(image, 1)
            cv2.imwrite(f"{output_dir}/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    return count

def prepare_validation_set(ref_database_name, que_database_name, ref_split, que_split, estimator_cfg):
    ref_database = parse_database_name(ref_database_name)
    que_database = parse_database_name(que_database_name)
    _, que_ids = get_database_split(que_database, que_split)

    estimator_cfg = load_cfg(estimator_cfg)
    estimator_cfg['refiner']=None
    estimator = Gen6DEstimator(estimator_cfg)
    estimator.build(ref_database, split_type=ref_split)

    img_id2det_info, img_id2sel_info = {}, {}
    for que_id in tqdm(que_ids):
        # estimate pose
        img = que_database.get_image(que_id)
        K = que_database.get_K(que_id)
        _, inter_results = estimator.predict(img, K)

        det_scale_r2q = inter_results['det_scale_r2q']
        det_position = inter_results['det_position']
        self_angle_r2q = inter_results['sel_angle_r2q']
        ref_idx = inter_results['sel_ref_idx']
        ref_pose = estimator.ref_info['poses'][ref_idx]
        ref_K = estimator.ref_info['Ks'][ref_idx]
        img_id2det_info[que_id]=(det_position, det_scale_r2q, 0)
        img_id2sel_info[que_id]=(self_angle_r2q, ref_pose, ref_K)

    save_pickle(img_id2det_info,f'data/val/det/{que_database_name}/{estimator.detector.cfg["name"]}.pkl')
    save_pickle(img_id2sel_info,f'data/val/sel/{que_database_name}/{estimator.detector.cfg["name"]}-{estimator.selector.cfg["name"]}.pkl')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True)

    # for video2image
    parser.add_argument('--input', type=str, default='example/video/mouse-ref.mp4')
    parser.add_argument('--output', type=str, default='example/mouse/images')
    parser.add_argument('--frame_inter', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=960)
    parser.add_argument('--transpose', action='store_true', dest='transpose', default=False)

    # for sfm
    parser.add_argument('--database_name', type=str, default='example/mouse')
    parser.add_argument('--colmap_path', type=str, default='colmap')
    # for sfm
    parser.add_argument('--que_database', type=str, default='linemod/cat')
    parser.add_argument('--que_split', type=str, default='linemod_test')
    parser.add_argument('--ref_database', type=str, default='linemod/cat')
    parser.add_argument('--ref_split', type=str, default='linemod_test')
    parser.add_argument('--estimator_cfg', type=str, default='configs/gen6d_train.yaml')
    args = parser.parse_args()

    if args.action == 'video2image':
        video2image(args.input,args.output,args.frame_inter,args.image_size, args.transpose)
    elif args.action=='sfm':
        build_colmap_model_no_pose(parse_database_name(args.database_name),args.colmap_path)
    elif args.action=='gen_val_set':
        prepare_validation_set(args.ref_database,args.que_database,args.ref_split,args.que_split,args.estimator_cfg)
    else:
        raise NotImplementedError