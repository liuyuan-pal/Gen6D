import random
import os

import numpy as np

from dataset.database import Co3D_ROOT
from utils.base_utils import read_pickle, save_pickle

####################google scan objects##########################
def get_gso_split(resolution=128):
    gso_split_pkl = 'data/gso_split.pkl'
    if os.path.exists(gso_split_pkl):
        train_fns, val_fns, test_fns = read_pickle(gso_split_pkl)
    else:
        if os.path.exists('data/google_scanned_objects'):
            sym_fns = np.loadtxt('assets/gso_sym.txt',dtype=np.str).tolist()
            gso_fns = []
            for fn in os.listdir('data/google_scanned_objects'):
                if os.path.isdir(os.path.join('data/google_scanned_objects',fn)) and fn not in sym_fns:
                    gso_fns.append(fn)

            random.seed(1234)
            random.shuffle(gso_fns)
            val_fns, test_fns, train_fns = gso_fns[:5], gso_fns[5:20], gso_fns[20:]
            save_pickle([train_fns, val_fns, test_fns], gso_split_pkl)
        else:
            val_fns, test_fns, train_fns = [], [], []

    gso_train_names = [f'gso/{fn}/white_{resolution}' for fn in train_fns]
    gso_val_names = [f'gso/{fn}/white_{resolution}' for fn in val_fns]
    gso_test_names = [f'gso/{fn}/white_{resolution}' for fn in test_fns]
    return gso_train_names, gso_val_names, gso_test_names

gso_train_names_128, gso_val_names_128, gso_test_names_128 = get_gso_split(128)

###############################Co3D###############################
# elevation >= 30 degree
# 32 sample points, max angle difference is 20 degree
# 1024 sample points, max angle difference is 3.5 degree
def get_co3d_category_split(category):
    seq_names_fn = f'{Co3D_ROOT}_256_512/{category}/valid_seq_names.pkl'
    seq_names = read_pickle(seq_names_fn)
    random.seed(1234)
    random.shuffle(seq_names)
    seq_names = [f'co3d_resize/{category}/{name}/256_512' for name in seq_names]
    train_names, val_names = seq_names[2:], seq_names[:2]
    return train_names, val_names

co3d_categories = np.loadtxt('assets/co3d_names.txt',dtype=np.str).tolist()

def get_co3d_split(category_num=None):
    if not os.path.exists(Co3D_ROOT) and not os.path.exists(f'{Co3D_ROOT}_256_512'): return [], []
    train_names, val_names = [], []
    cur_co3d_categories = [item for item in co3d_categories]
    for c in cur_co3d_categories:
        ts, vs = get_co3d_category_split(c)
        if category_num is None:
            train_names += ts
        else:
            train_names += ts[:category_num]
        val_names += vs

    random.seed(1234)
    random.shuffle(val_names)
    return train_names, val_names[:10]

co3d_train_names, co3d_val_names = get_co3d_split()

###########################ShapeNet###############################
shapenet_excluded_clasees=[
    '02747177',
    '02876657',
    '02880940',
    '02808440',
    '04225987',
]
shapenet_excluded_instance=np.loadtxt('assets/shapenet_sym_objects.txt', dtype=np.str).tolist()
shapenet_train_names = read_pickle(f'data/shapenet/shapenet_render_v1.pkl')

# 'co3d_train', 'gso_train_128', 'shapenet_train_v1', 'linemod_train', 'gen6d_train'
name2database_names={
    'gso_train_128': gso_train_names_128,
    'co3d_train': co3d_train_names,
    'shapenet_train': shapenet_train_names,
    'linemod_train': [f'linemod/{obj}' for obj in ['ape','can','holepuncher','iron','phone']],
    'genmop_train': [f'genmop/{name}-test' for name in ['cup','knife','love','plug_cn','miffy']],

    'gso_train_128_exp': gso_train_names_128[:10],
    'co3d_train_exp': co3d_train_names[:10],
    'shapenet_train_exp': shapenet_train_names[:10],
}