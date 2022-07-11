import argparse
from pathlib import Path

import cv2

from colmap_script import build_colmap_model_no_pose
from dataset.database import parse_database_name


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
    args = parser.parse_args()

    if args.action == 'video2image':
        video2image(args.input,args.output,args.frame_inter,args.image_size, args.transpose)
    elif args.action=='sfm':
        extractor = None
        matcher = None
        build_colmap_model_no_pose(parse_database_name(args.database_name),args.colmap_path)
    else:
        raise NotImplementedError