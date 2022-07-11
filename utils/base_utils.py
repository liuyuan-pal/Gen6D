import math
import os

import cv2
import plyfile
import torch

import numpy as np
import pickle

import yaml
from numpy import ndarray
from plyfile import PlyData
from skimage.io import imread

from torch import Tensor
from transforms3d.euler import euler2mat

#########################IO#####################################

def load_point_cloud(pcl_path):
    with open(pcl_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)
        xyz = np.stack([np.array(plydata["vertex"][c]).astype(float) for c in ("x", "y", "z")], axis=1)
    return xyz

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def compute_precision_recall_np(pr,gt,eps=1e-5):
    tp=np.sum(gt & pr)
    fp=np.sum((~gt) & pr)
    fn=np.sum(gt & (~pr))
    precision=(tp+eps)/(fp+tp+eps)
    recall=(tp+eps)/(tp+fn+eps)
    if precision<1e-3 or recall<1e-3:
        f1=0.0
    else:
        f1=(2*precision*recall+eps)/(precision+recall+eps)

    return precision, recall, f1

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_stem(path,suffix_len=5):
    return os.path.basename(path)[:-suffix_len]

def load_component(component_func,component_cfg_fn):
    component_cfg=load_cfg(component_cfg_fn)
    return component_func[component_cfg['type']](component_cfg)

def load_ply_model(model_path):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    return np.stack([x, y, z], axis=-1)

def save_depth(fn,depth,max_val=1000):
    import png
    depth = np.clip(depth,a_min=0,a_max=max_val)/max_val*65535
    depth = depth.astype(np.uint16)
    with open(fn, 'wb') as f:
        writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16, greyscale=True)
        zgray2list = depth.tolist()
        writer.write(f, zgray2list)

#####################depth and image###############################

def mask_zbuffer_to_pts(mask, zbuffer, K):
    ys,xs=np.nonzero(mask)
    zbuffer=zbuffer[ys, xs]
    u,v,f=K[0,2],K[1,2],K[0,0]
    depth = zbuffer / np.sqrt((xs - u + 0.5) ** 2 + (ys - v + 0.5) ** 2 + f ** 2) * f

    pts=np.asarray([xs, ys, depth], np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    return np.dot(pts,np.linalg.inv(K).transpose())

def mask_depth_to_pts(mask,depth,K,rgb=None):
    hs,ws=np.nonzero(mask)
    depth=depth[hs,ws]
    pts=np.asarray([ws,hs,depth],np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    if rgb is not None:
        return np.dot(pts, np.linalg.inv(K).transpose()), rgb[hs,ws]
    else:
        return np.dot(pts, np.linalg.inv(K).transpose())

def read_render_zbuffer(dpt_pth,max_depth,min_depth):
    zbuffer = imread(dpt_pth)
    mask = (zbuffer>0) & (zbuffer<5000)
    zbuffer=zbuffer.astype(np.float64)/2**16*(max_depth-min_depth)+min_depth
    return mask, zbuffer

def zbuffer_to_depth(zbuffer,K):
    u,v,f=K[0,2],K[1,2],K[0,0]
    x=np.arange(zbuffer.shape[1])
    y=np.arange(zbuffer.shape[0])
    x,y=np.meshgrid(x,y)
    x=np.reshape(x,[-1,1])
    y=np.reshape(y,[-1,1])
    depth = np.reshape(zbuffer,[-1,1])

    depth = depth / np.sqrt((x - u + 0.5) ** 2 + (y - v + 0.5) ** 2 + f ** 2) * f
    return np.reshape(depth,zbuffer.shape)

def color_map_forward(rgb):
    return rgb.astype(np.float32)/255

def color_map_backward(rgb):
    rgb=rgb*255
    rgb=np.clip(rgb,a_min=0,a_max=255).astype(np.uint8)
    return rgb

def rotate_image(rot, pose, K, img, mask):
    if isinstance(rot, np.ndarray):
        R = rot
    else:
        R=np.array([[np.cos(rot), -np.sin(rot), 0.0],
                    [np.sin(rot), np.cos(rot), 0.0],
                    [0,0,1]],dtype=np.float32)

    # adjust pose
    pose_adj=np.copy(pose)
    pose_adj[:, :3] = R @ pose_adj[:, :3]
    pose_adj[:, 3:] = R @ pose_adj[:, 3:]

    # adjust image
    transform = K @ R @ np.linalg.inv(K) # transform original
    h, w, _ = img.shape

    ys, xs = np.nonzero(mask)
    coords = np.stack([xs,ys],-1).astype(np.float32)
    coords_new = cv2.perspectiveTransform(coords[:,None,:],transform)[:,0,:]
    x_min, y_min = np.floor(np.min(coords_new,0)).astype(np.int32)
    x_max, y_max = np.ceil(np.max(coords_new,0)).astype(np.int32)
    th, tw = y_max - y_min, x_max - x_min
    translation = np.identity(3)
    translation[0,2]=-x_min
    translation[1,2]=-y_min
    K = translation @ K

    transform = translation @ transform
    img = cv2.warpPerspective(img, transform, (tw, th), flags=cv2.INTER_LINEAR)
    return img, pose_adj, K

def resize_img(img, ratio):
    # if ratio>=1.0: return img
    h, w, _ = img.shape
    hn, wn = int(np.round(h * ratio)), int(np.round(w * ratio))
    img_out = cv2.resize(downsample_gaussian_blur(img, ratio), (wn, hn), cv2.INTER_LINEAR)
    return img_out

def pad_img(img,padding_interval=8):
    h, w = img.shape[:2]
    hp = (padding_interval - (h % padding_interval)) % padding_interval
    wp = (padding_interval - (w % padding_interval)) % padding_interval
    if hp != 0 or wp != 0:
        img = np.pad(img, ((0, hp), (0, wp), (0, 0)), 'edge')
    return img

def pad_img_end(img,th,tw,padding_mode='edge',constant_values=0):
    h, w = img.shape[:2]
    hp = th-h
    wp = tw-w
    if hp != 0 or wp != 0:
        if padding_mode=='constant':
            img = np.pad(img, ((0, hp), (0, wp), (0, 0)), padding_mode, constant_values=constant_values)
        else:
            img = np.pad(img, ((0, hp), (0, wp), (0, 0)), padding_mode)
    return img

def pad_img_target(img, th, tw, K=np.eye(3), background_color=0):
    h, w = img.shape[:2]
    hp = th - h
    wp = tw - w
    if hp != 0 or wp != 0:
        if len(img.shape) == 3:
            img = np.pad(img, ((hp//2, hp-hp//2), (wp//2, wp - wp//2), (0, 0)), 'constant', constant_values=background_color)
        elif len(img.shape) == 2:
            img = np.pad(img, ((hp // 2, hp - hp // 2), (wp // 2, wp - wp // 2)), 'constant', constant_values=background_color)
        else:
            print(f'image shape unknown {img.shape}')
            raise NotImplementedError
        translation = np.identity(3)
        translation[0,2]=wp//2
        translation[1,2]=hp//2
        K = translation @ K
    return img, K

#######################image processing#############################

def grey_repeats(img_raw):
    if len(img_raw.shape) == 2: img_raw = np.repeat(img_raw[:, :, None], 3, axis=2)
    if img_raw.shape[2] > 3: img_raw = img_raw[:, :, :3]
    return img_raw

def normalize_image(img,mask=None):
    if mask is not None: img[np.logical_not(mask.astype(np.bool))]=127
    img=(img.transpose([2,0,1]).astype(np.float32)-127.0)/128.0
    return torch.tensor(img,dtype=torch.float32)

def tensor_to_image(tensor):
    return (tensor * 128 + 127).astype(np.uint8).transpose(1,2,0)

def equal_hist(img):
    if len(img.shape)==3:
        img0=cv2.equalizeHist(img[:,:,0])
        img1=cv2.equalizeHist(img[:,:,1])
        img2=cv2.equalizeHist(img[:,:,2])
        img=np.concatenate([img0[...,None],img1[...,None],img2[...,None]],2)
    else:
        img=cv2.equalizeHist(img)
    return img

def resize_large_image(img,resize_max):
    h,w=img.shape[:2]
    max_side = max(h, w)
    if max_side > resize_max:
        ratio = resize_max / max_side
        if ratio <= 0.5: img = cv2.GaussianBlur(img, (5, 5), 1.5)
        img = cv2.resize(img, (int(round(ratio * w)), int(round(ratio * h))), interpolation=cv2.INTER_LINEAR)
        return img, ratio
    else:
        return img, 1.0

def downsample_gaussian_blur(img,ratio):
    sigma=(1/ratio)/3
    # ksize=np.ceil(2*sigma)
    ksize=int(np.ceil(((sigma-0.8)/0.3+1)*2+1))
    ksize=ksize+1 if ksize%2==0 else ksize
    img=cv2.GaussianBlur(img,(ksize,ksize),sigma,borderType=cv2.BORDER_REFLECT101)
    return img

def resize_small_image(img,resize_min):
    h,w=img.shape[:2]
    min_side = min(h, w)
    if min_side < resize_min:
        ratio = resize_min / min_side
        img = cv2.resize(img, (int(round(ratio * w)), int(round(ratio * h))), interpolation=cv2.INTER_LINEAR)
        return img, ratio
    else:
        return img, 1.0
    
############################geometry######################################
def project_points(pts,RT,K):
    pts = np.matmul(pts,RT[:,:3].transpose())+RT[:,3:].transpose()
    pts = np.matmul(pts,K.transpose())
    dpt = pts[:,2]
    mask0 = (np.abs(dpt)<1e-4) & (np.abs(dpt)>0)
    if np.sum(mask0)>0: dpt[mask0]=1e-4
    mask1=(np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1)>0: dpt[mask1]=-1e-4
    pts2d = pts[:,:2]/dpt[:,None]
    return pts2d, dpt

def round_coordinates(coord,h,w):
    coord=np.round(coord).astype(np.int32)
    coord[coord[:,0]<0,0]=0
    coord[coord[:,0]>=w,0]=w-1
    coord[coord[:,1]<0,1]=0
    coord[coord[:,1]>=h,1]=h-1
    return coord

def perspective_transform(pts, H):
    tpts = np.concatenate([pts, np.ones([pts.shape[0], 1])], 1) @ H.transpose()
    tpts = tpts[:, :2] / np.abs(tpts[:, 2:]) # todo: why only abs? this one is correct
    return tpts

def get_rot_m(angle):
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32) # rn+1,3,3

def get_rot_m_batch(angle):
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32).transpose([2,0,1])

def compute_F(K1, K2, R, t):
    """

    :param K1: [3,3]
    :param K2: [3,3]
    :param R:  [3,3]
    :param t:  [3,1]
    :return:
    """
    A = K1 @ R.T @ t # [3,1]
    C = np.asarray([[0,-A[2,0],A[1,0]],
                    [A[2,0],0,-A[0,0]],
                    [-A[1,0],A[0,0],0]])
    F = (np.linalg.inv(K2)).T @ R @ K1.T @ C
    return F

def compute_relative_transformation(Rt0,Rt1):
    """
    x1=Rx0+t
    :param Rt0: x0=R0x+t0
    :param Rt1: x1=R1x+t1
    :return:
        R1R0.T(x0-t0)+t1
    """
    R=Rt1[:,:3] @ Rt0[:,:3].T
    t=Rt1[:,3] - R @ Rt0[:,3]
    return np.concatenate([R,t[:,None]],1)

def pts_to_hpts(pts):
    return np.concatenate([pts,np.ones([pts.shape[0],1])],1)

def hpts_to_pts(hpts):
    return hpts[:,:-1]/hpts[:,-1:]

def np_skew_symmetric(v):
    M = np.asarray([
        [0, -v[2], v[1],],
        [v[2], 0, -v[0],],
        [-v[1], v[0], 0,],
    ])

    return M

def point_line_dist(hpts,lines):
    """
    :param hpts: n,3 or n,2
    :param lines: n,3
    :return:
    """
    if hpts.shape[1]==2:
        hpts=np.concatenate([hpts,np.ones([hpts.shape[0],1])],1)
    return np.abs(np.sum(hpts*lines,1))/np.linalg.norm(lines[:,:2],2,1)

def epipolar_distance(x0, x1, F):
    """

    :param x0: [n,2]
    :param x1: [n,2]
    :param F:  [3,3]
    :return:
    """

    hkps0 = np.concatenate([x0, np.ones([x0.shape[0], 1])], 1)
    hkps1 = np.concatenate([x1, np.ones([x1.shape[0], 1])], 1)

    lines1 = hkps0 @ F.T
    lines0 = hkps1 @ F

    dist10 = point_line_dist(hkps0, lines0)
    dist01 = point_line_dist(hkps1, lines1)

    return dist10, dist01

def epipolar_distance_mean(x0, x1, F):
    return np.mean(np.stack(epipolar_distance(x0,x1,F),1),1)

def compute_dR_dt(R0, t0, R1, t1):
    # Compute dR, dt
    dR = np.dot(R1, R0.T)
    dt = t1 - np.dot(dR, t0)
    return dR, dt

def interpolate_image_points(img, pts, interpolation=cv2.INTER_LINEAR):
    # img [h,w,k] pts [n,2]
    if len(pts)<32767:
        pts=pts.astype(np.float32)
        return cv2.remap(img,pts[:,None,0],pts[:,None,1],borderMode=cv2.BORDER_CONSTANT,borderValue=0,interpolation=interpolation)[:,0]
        # pn=len(pts)
        # sl=int(np.ceil(np.sqrt(pn)))
        # tmp_img=np.zeros([sl*sl,2],np.float32)
        # tmp_img[:pn]=pts
        # tmp_img=tmp_img.reshape([sl,sl,2])
        # tmp_img=cv2.remap(img,tmp_img[:,:,0],tmp_img[:,:,1],borderMode=cv2.BORDER_CONSTANT,borderValue=0,interpolation=interpolation)
        # return tmp_img.flatten()[:pn]
    else:
        results=[]
        for k in range(0,len(pts),30000):
            results.append(interpolate_image_points(img, pts[k:k + 30000], interpolation))
        return np.concatenate(results,0)

def transform_points_Rt(pts, R, t):
    t = t.flatten()
    return pts @ R.T + t[None,:]

def transform_points_pose(pts, pose):
    R, t = pose[:, :3], pose[:, 3]
    if len(pts.shape)==1:
        return (R @ pts[:,None] + t[:,None])[:,0]
    return pts @ R.T + t[None,:]

def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q

def compute_rotation_angle_diff(R_gt,R):
    eps = 1e-15
    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)
    return np.rad2deg(np.abs(err_q))

def compute_translation_angle_diff(t_gt,t):
    eps=1e-15
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))
    return np.rad2deg(np.abs(err_t))

def get_identity_pose():
    return np.concatenate([np.identity(3),np.zeros([3,1])],1).astype(np.float32)

def pose_inverse(pose):
    R = pose[:,:3].T
    t = - R @ pose[:,3:]
    return np.concatenate([R,t],-1)

def similarity_pose_inverse(pose):
    A = np.linalg.inv(pose[:,:3])
    t = - A @ pose[:,3:]
    return np.concatenate([A,t],-1)

def pose_compose(pose0, pose1):
    """
    apply pose0 first, then pose1
    :param pose0:
    :param pose1:
    :return:
    """
    t = pose1[:,:3] @ pose0[:,3:] + pose1[:,3:]
    R = pose1[:,:3] @ pose0[:,:3]
    return np.concatenate([R,t], 1)

def pose_apply(pose,pts):
    return transform_points_pose(pts, pose)

def to_cuda(data):
    if type(data)==list:
        results = []
        for i, item in enumerate(data):
            results.append(to_cuda(item))
        return results
    elif type(data)==dict:
        results={}
        for k,v in data.items():
            results[k]=to_cuda(v)
        return results
    elif type(data).__name__ == "Tensor":
        return data.cuda()
    else:
        return data

def to_cpu_numpy(data):
    if type(data)==list:
        results = []
        for i, item in enumerate(data):
            results.append(to_cpu_numpy(item))
        return results
    elif type(data)==dict:
        results={}
        for k,v in data.items():
            results[k]=to_cpu_numpy(v)
        return results
    elif type(data).__name__ == "Tensor":
        return data.detach().cpu().numpy()
    else:
        return data

def sample_fps_points(points, sample_num, init_center=True, index_model=False, init_first=False, init_first_index=0, init_point=None):
    sample_num = min(points.shape[0],sample_num)
    output_index=[]
    if init_point is None:
        if init_center:
            init_point=np.mean(points,0)
        else:
            if init_first:
                init_index=init_first_index
            else:
                init_index=np.random.randint(0, points.shape[0])
            init_point=points[init_index]
            output_index.append(init_index)

    output_points=[init_point]
    cur_point=init_point
    distance=np.full(points.shape[0],1e8)
    for k in range(min(sample_num-1, points.shape[0]-1)):
        cur_distance=np.linalg.norm(cur_point[None,:]-points,2,1)
        distance=np.min(np.stack([cur_distance,distance],1),1)
        cur_index=np.argmax(distance)
        cur_point=points[cur_index]
        output_points.append(cur_point)
        output_index.append(cur_index)

    if index_model:
        return np.asarray(output_index)
    else:
        return np.asarray(output_points)

def triangulate(kps0,kps1,pose0,pose1,K0,K1):
    kps0_ = hpts_to_pts(pts_to_hpts(kps0) @ np.linalg.inv(K0).T)
    kps1_ = hpts_to_pts(pts_to_hpts(kps1) @ np.linalg.inv(K1).T)
    pts3d = cv2.triangulatePoints(pose0.astype(np.float64),pose1.astype(np.float64),
                                  kps0_.T.astype(np.float64),kps1_.T.astype(np.float64)).T
    pts3d = pts3d[:,:3]/pts3d[:,3:]
    return pts3d

def transformation_inverse_2d(trans):
    A=trans[:2,:2] # 2,2
    t=trans[:,2:]  # 2,1
    if isinstance(trans, Tensor):
        A_ = np.linalg.inv(A)
        t_ = -A_ @ t
        return torch.cat([A_, t_], 1)
    elif isinstance(trans, ndarray):
        A_ = np.linalg.inv(A)
        t_ = -A_ @ t
        return np.concatenate([A_, t_], 1)
    else:
        raise NotImplementedError

def transformation_compose_2d(trans0, trans1):
    """
    @param trans0: [2,3]
    @param trans1: [2,3]
    @return: apply trans0 then trans1
    """
    t1 = trans1[:, 2]
    t0 = trans0[:, 2]
    R1 = trans1[:, :2]
    R0 = trans0[:, :2]
    R = R1 @ R0
    t = R1 @ t0 + t1
    return np.concatenate([R, t[:, None]], 1)

def transformation_apply_2d(trans,points):
    return points @ trans[:,:2].T + trans[:,2:].T

def angle_to_rotation_2d(angle):
    return np.asarray([[np.cos(angle),-np.sin(angle)],
                       [np.sin(angle),np.cos(angle)]])

def transformation_offset_2d(x,y):
    return np.concatenate([np.eye(2),np.asarray([x,y])[:,None]],1).astype(np.float32)

def transformation_scale_2d(scale):
    return np.concatenate([np.diag([scale,scale]),np.zeros([2,1])],1).astype(np.float32)

def transformation_rotation_2d(ang):
    return np.concatenate([angle_to_rotation_2d(ang),np.zeros([2,1])],1).astype(np.float32)

def transformation_decompose_2d(M):
    scale = np.sqrt(np.linalg.det(M[:2, :2]))
    rotation = np.arctan2(M[1, 0], M[0, 0])
    offset = M[:2,2]
    return scale, rotation, offset

def transformation_crop(img, position, scale, angle, size, new_position=None):
    M = transformation_offset_2d(-position[0], -position[1])
    M = transformation_compose_2d(M, transformation_scale_2d(scale))
    M = transformation_compose_2d(M, transformation_rotation_2d(angle))
    if new_position is None:
        M = transformation_compose_2d(M, transformation_offset_2d(size / 2, size / 2))
    else:
        M = transformation_compose_2d(M, transformation_offset_2d(new_position[0], new_position[1]))
    img_region = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)
    return img_region, M

def look_at_rotation(point):
    """
    @param point: point in normalized image coordinate not in pixels
    @return: R
    R @ x_raw -> x_lookat
    """
    x, y = point
    R1 = euler2mat(-np.arctan2(x, 1),0,0,'syxz')
    R2 = euler2mat(np.arctan2(y, 1),0,0,'sxyz')
    return R2 @ R1
