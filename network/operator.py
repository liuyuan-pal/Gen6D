import numpy as np
import torch

def normalize_coords(coords: torch.Tensor, h, w):
    """
    normalzie coords to [-1,1]
    @param coords:
    @param h:
    @param w:
    @return:
    """
    coords = torch.clone(coords)
    coords = coords + 0.5
    coords[...,0] = coords[...,0]/w
    coords[...,1] = coords[...,1]/h
    coords = (coords - 0.5)*2
    return coords

def pose_apply_th(poses,pts):
    return pts @ poses[:,:,:3].permute(0,2,1) + poses[:,:,3:].permute(0,2,1)

def generate_coords(h,w,device):
    coords=torch.stack(torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device)),-1)
    return coords[...,(1,0)]