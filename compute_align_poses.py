import xml.etree.cElementTree as ET

from pathlib import Path

import cv2
from skimage.io import imread, imsave

from dataset.database import BaseDatabase, GenMOP_ROOT, parse_colmap_project, parse_database_name
from utils.base_utils import read_pickle, hpts_to_pts, pts_to_hpts, save_pickle
from utils.draw_utils import draw_keypoints, concat_images_list
from utils.read_write_model import read_model

import numpy as np

import os

def triangulate(kps0,kps1,pose0,pose1,K0,K1):
    kps0_ = hpts_to_pts(pts_to_hpts(kps0) @ np.linalg.inv(K0).T)
    kps1_ = hpts_to_pts(pts_to_hpts(kps1) @ np.linalg.inv(K1).T)
    pts3d = cv2.triangulatePoints(pose0.astype(np.float64),pose1.astype(np.float64),
                                  kps0_.T.astype(np.float64),kps1_.T.astype(np.float64)).T
    pts3d = pts3d[:,:3]/pts3d[:,3:]
    return pts3d

class GenMOPCOLMAPDatabase(BaseDatabase):
    """
    this class simply read the colmap project, not align them
    """
    def __init__(self, database_name):
        super().__init__(database_name)

        _, seq_name = database_name.split('/') # genmop/object_name-test or genmop/object_name-ref

        # get image filenames
        self.seq_name = seq_name
        self.root = Path(GenMOP_ROOT) / self.seq_name
        img_fns_cache = self.root / 'images_fn_cache.pkl'
        self.img_fns = read_pickle(str(img_fns_cache))

        # parse colmap project
        cameras, images, points3d = read_model(f'{GenMOP_ROOT}/{seq_name}/colmap-all/colmap_default-colmap_default/sparse/0')
        self.poses, self.Ks, self.img_ids = parse_colmap_project(cameras, images, self.img_fns)

    def get_image(self, img_id):
        return imread(str(self.root / 'images' / self.img_fns[int(img_id)]))

    def get_K(self, img_id):
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids

def _parse_fad(fn,):
    tree = ET.ElementTree(file=fn)
    root = tree.getroot()
    features = root[0][0][0]
    keypoints = []
    for feature in features:
        x = int(feature.attrib['x'])
        y = int(feature.attrib['y'])
        keypoints.append((x, y))
    return keypoints

def align(ref_database, test_database, input_dir):
    test_anno, ref_anno = [], []
    for fn in os.listdir(input_dir):
        fn_name = fn.split('-')[1]
        kps = _parse_fad(f'{input_dir}/{fn}')
        if fn.startswith('test') or fn.startswith('set'):
            test_anno.append({'name': fn_name + '.jpg', 'kps': kps})
        if fn.startswith('ref'):
            ref_anno.append({'name': fn_name + '.jpg', 'kps': kps})

    assert (len(ref_anno) == 2)
    assert (len(test_anno) == 2)
    ref_id0 = str(ref_database.img_fns.index(ref_anno[0]['name']))
    ref_id1 = str(ref_database.img_fns.index(ref_anno[1]['name']))
    test_id0 = str(test_database.img_fns.index(test_anno[0]['name']))
    test_id1 = str(test_database.img_fns.index(test_anno[1]['name']))

    def triangulation_from_annotations(database, annotation, id0, id1):
        pose0 = database.get_pose(id0)
        pose1 = database.get_pose(id1)
        K0 = database.get_K(id0)
        K1 = database.get_K(id1)
        kps0 = np.asarray(annotation[0]['kps'])
        kps1 = np.asarray(annotation[1]['kps'])
        pts3d = triangulate(kps0, kps1, pose0, pose1, K0, K1)
        return pts3d

    # triangulation
    pts3d_ref = triangulation_from_annotations(ref_database, ref_anno, ref_id0, ref_id1)
    pts3d_test = triangulation_from_annotations(test_database, test_anno, test_id0, test_id1)

    pts_test = (pts3d_test - np.mean(pts3d_test, 0))
    pts_ref = (pts3d_ref - np.mean(pts3d_ref, 0))
    norm_test = np.linalg.norm(pts_test, 2, 1)
    norm_ref = np.linalg.norm(pts_ref, 2, 1)
    transfer_scale = np.mean(norm_test / norm_ref)
    pts_ref *= transfer_scale
    U, S, VT = np.linalg.svd(pts_ref.T @ pts_test)
    R = VT.T @ U.T
    t = np.mean(pts3d_test, 0)[:, None] - transfer_scale * (R @ np.mean(pts3d_ref, 0)[:, None])
    transfer_pose = np.concatenate([R, t], 1)  # x_new to x_old
    transfer_scale = 1/transfer_scale
    return transfer_scale, transfer_pose


if __name__=="__main__":
    # we already provide the manual annotated keypoints in 'align-data/tformer-anno'
    # 'ref-frame40' means the labeled keypoints on frame40 of the reference video.
    # 'test-frame130' means the labeled keypoints on frame130 of the test(query) video.
    input_dir = 'align-data/tformer-anno'
    # note this will output the align poses, which will be used in GenMOPDatabase
    output_fn = f'{GenMOP_ROOT}/tformer-test/align.pkl'
    ref_database = GenMOPCOLMAPDatabase('genmop/tformer-ref')
    test_database = GenMOPCOLMAPDatabase('genmop/tformer-test')


    # let's first visualize the keypoints

    # read annotations and find the corresponding image ids
    test_anno, ref_anno = [], []
    for fn in os.listdir(input_dir):
        fn_name = fn.split('-')[1]
        kps = _parse_fad(f'{input_dir}/{fn}')
        if fn.startswith('test') or fn.startswith('set'):
            test_anno.append({'name': fn_name + '.jpg', 'kps': kps})
        if fn.startswith('ref'):
            ref_anno.append({'name': fn_name + '.jpg', 'kps': kps})
    assert (len(ref_anno) == 2)
    assert (len(test_anno) == 2)
    ref_id0 = str(ref_database.img_fns.index(ref_anno[0]['name']))
    ref_id1 = str(ref_database.img_fns.index(ref_anno[1]['name']))
    test_id0 = str(test_database.img_fns.index(test_anno[0]['name']))
    test_id1 = str(test_database.img_fns.index(test_anno[1]['name']))

    ref_kps0, ref_kps1 = np.asarray(ref_anno[0]['kps']), np.asarray(ref_anno[1]['kps'])
    test_kps0, test_kps1 = np.asarray(test_anno[0]['kps']), np.asarray(test_anno[1]['kps'])

    ref_img0, ref_img1 = ref_database.get_image(ref_id0), ref_database.get_image(ref_id1)
    test_img0, test_img1 = test_database.get_image(test_id0), test_database.get_image(test_id1)
    colors = np.asarray([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)],np.uint8)
    imsave('data/vis-tformer-anno.jpg', # note: output image will be saved in this path
           concat_images_list(draw_keypoints(ref_img0, ref_kps0, colors),
                              draw_keypoints(ref_img1, ref_kps1, colors),
                              draw_keypoints(test_img0, test_kps0, colors),
                              draw_keypoints(test_img1, test_kps1, colors),
                              )
           )

    # actual codes to compute the alignment
    transfer_scale, transfer_pose = align(ref_database, test_database, input_dir)


    # you can see that the poses and scale from the 'tformer-test/align.pkl' are the same as the computed `transfer_pose` and `transfer scale`
    transfer_scale_, transfer_pose_ = read_pickle(output_fn)
    print(transfer_pose_)
    print(transfer_pose)
    print(transfer_scale_)
    print(transfer_scale)

    # uncomment the following line, you may save the results
    # save_pickle((transfer_scale, transfer_pose), output_fn)
