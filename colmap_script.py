import logging
import subprocess
import os
from pathlib import Path

import numpy as np
from skimage.io import imsave

from dataset.database import BaseDatabase, get_database_split
from utils.colmap_database import COLMAPDatabase
from utils.read_write_model import CAMERA_MODEL_NAMES

def run_sfm(colmap_path, model_path, database_path, image_dir):
    logging.info('Running the triangulation...')
    model_path.mkdir(exist_ok=True, parents=True)

    cmd = [
        str(colmap_path), 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(model_path),
    ]
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)

def run_patch_match(colmap_path, sparse_model: Path, image_dir: Path, dense_model: Path):
    logging.info('Running patch match...')
    assert sparse_model.exists()
    dense_model.mkdir(parents=True, exist_ok=True)
    cmd = [str(colmap_path), 'image_undistorter', '--input_path', str(sparse_model), '--image_path', str(image_dir), '--output_path', str(dense_model),]
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)
    cmd = [str(colmap_path), 'patch_match_stereo','--workspace_path', str(dense_model),]
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)

def run_depth_fusion(colmap_path, dense_model: Path, ply_path: Path):
    logging.info('Running patch match...')
    dense_model.mkdir(parents=True, exist_ok=True)
    cmd = [str(colmap_path), 'stereo_fusion',
           '--workspace_path', str(dense_model),
           '--workspace_format', 'COLMAP',
           '--input_type', 'geometric',
           '--output_path', str(ply_path),]
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)

def dump_images(database, ref_ids, image_path: Path):
    image_path.mkdir(parents=True, exist_ok=True)
    for ref_id in ref_ids:
        if (image_path / f'{ref_id}.jpg').exists():
            continue
        else:
            imsave(str(image_path / f'{ref_id}.jpg'),database.get_image(ref_id))

def extract_and_match_sift(colmap_path, database_path, image_dir):
    cmd = [
        str(colmap_path), 'feature_extractor',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
    ]
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)
    cmd = [
        str(colmap_path), 'exhaustive_matcher',
        '--database_path', str(database_path),
    ]
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)

def create_db_from_database(database, ref_ids, database_path: Path):
    if database_path.exists():
        logging.warning('Database already exists. we will skip db creation.')
        return

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for ri, ref_id in enumerate(ref_ids):
        img = database.get_image(ref_id)
        h, w = img.shape[:2]
        model_id = CAMERA_MODEL_NAMES["SIMPLE_RADIAL"].model_id
        db.add_camera(model_id, float(w), float(h), np.asarray([np.sqrt(h**2+w**2), w/2.0, h/2.0, 0.0],np.float64), camera_id=ri+1)
        db.add_image(f'{ref_id}.jpg', ri+1, image_id=ri+1)

    db.commit()
    db.close()

def build_colmap_model_no_pose(database: BaseDatabase, colmap_path='colmap'):
    colmap_root = Path('data') / database.database_name / 'colmap'
    colmap_root.mkdir(exist_ok=True, parents=True)
    image_path = colmap_root / 'images'
    database_path = colmap_root / 'database.db'

    ref_ids, _ = get_database_split(database, 'all')

    dump_images(database, ref_ids, image_path)
    create_db_from_database(database, ref_ids, database_path)
    extract_and_match_sift(colmap_path, database_path, image_path)

    sparse_model_path = colmap_root / f'sparse'
    dense_model_path = colmap_root / f'dense'
    ply_path = colmap_root / f'pointcloud.ply'
    run_sfm(colmap_path, sparse_model_path, database_path, image_path)
    run_patch_match(colmap_path, sparse_model_path / '0', image_path, dense_model_path)
    run_depth_fusion(colmap_path, dense_model_path, ply_path)

def clean_colmap_project(database, split_name):
    extractor_name = 'colmap_default'
    matcher_name = 'colmap_default'

    colmap_root = Path('data/colmap_projects') / database.database_name / f'colmap-{split_name}' / f'{extractor_name}-{matcher_name}'
    image_path = colmap_root / 'images'
    database_path = colmap_root / 'database.db'
    empty_model_path = colmap_root / 'empty'
    sparse_model_path = colmap_root / f'sparse'
    dense_model_path = colmap_root / f'dense'

    os.system(f'rm {str(sparse_model_path)} -r')
    os.system(f'rm {str(database_path)} -r')
    os.system(f'rm {str(image_path)} -r')
    os.system(f'rm {str(empty_model_path)} -r')
    os.system(f'rm {str(dense_model_path / "images")} -r')
    os.system(f'rm {str(dense_model_path / "sparse")} -r')
    os.system(f'rm {str(dense_model_path / "stereo" / "normal_maps")} -r')
    os.system(f'rm {str(dense_model_path / "stereo" / "depth_maps")}/*.photometric.bin')


