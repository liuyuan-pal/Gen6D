import numpy as np
import time
import random
import torch

def dummy_collate_fn(data_list):
    return data_list[0]

def simple_collate_fn(data_list):
    ks=data_list[0].keys()
    outputs={k:[] for k in ks}
    for k in ks:
        if isinstance(data_list[0][k], dict):
            outputs[k] = {k_: [] for k_ in data_list[0][k].keys()}
            for k_ in data_list[0][k].keys():
                for data in data_list:
                    outputs[k][k_].append(data[k][k_])
                outputs[k][k_]=torch.stack(outputs[k][k_], 0)
        else:
            for data in data_list:
                outputs[k].append(data[k])
            if isinstance(data_list[0][k], torch.Tensor):
                outputs[k]=torch.stack(outputs[k],0)
    return outputs

def set_seed(index,is_train):
    if is_train:
        np.random.seed((index+int(time.time()))%(2**16))
        random.seed((index+int(time.time()))%(2**16)+1)
        torch.random.manual_seed((index+int(time.time()))%(2**16)+1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)