import os
from collections import OrderedDict

import torch
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn as nn


def load_model(model, optim, model_dir, epoch=-1):
    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch

    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    model.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    print('load {} epoch {}'.format(model_dir, pretrained_model['epoch'] + 1))
    return pretrained_model['epoch'] + 1

def adjust_learning_rate(optimizer, epoch, lr_decay_rate, lr_decay_epoch, min_lr=1e-5):
    if ((epoch + 1) % lr_decay_epoch) != 0:
        return

    for param_group in optimizer.param_groups:
        # print(param_group)
        lr_before = param_group['lr']
        param_group['lr'] = param_group['lr'] * lr_decay_rate
        param_group['lr'] = max(param_group['lr'], min_lr)

    print('changing learning rate {:5f} to {:.5f}'.format(lr_before, max(param_group['lr'], min_lr)))

def reset_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        # print(param_group)
        # lr_before = param_group['lr']
        param_group['lr'] = lr
    # print('changing learning rate {:5f} to {:.5f}'.format(lr_before,lr))
    return lr

def save_model(net, optim, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.feats_state_dict(),
        'optim': optim.feats_state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))

class Recorder(object):
    def __init__(self, rec_dir, rec_fn):
        self.rec_dir = rec_dir
        self.rec_fn = rec_fn
        self.data = OrderedDict()
        self.writer = SummaryWriter(log_dir=rec_dir)

    def rec_loss(self, losses_batch, step, epoch, prefix='train', dump=False):
        for k, v in losses_batch.items():
            name = '{}/{}'.format(prefix, k)
            if name in self.data:
                self.data[name].append(v)
            else:
                self.data[name] = [v]

        if dump:
            if prefix == 'train':
                msg = '{} epoch {} step {} '.format(prefix, epoch, step)
            else:
                msg = '{} epoch {} '.format(prefix, epoch)
            for k, v in self.data.items():
                if not k.startswith(prefix): continue
                if len(v) > 0:
                    msg += '{} {:.5f} '.format(k.split('/')[-1], np.mean(v))
                    self.writer.add_scalar(k, np.mean(v), step)
                self.data[k] = []

            print(msg)
            with open(self.rec_fn, 'a') as f:
                f.write(msg + '\n')

    def rec_msg(self, msg):
        print(msg)
        with open(self.rec_fn, 'a') as f:
            f.write(msg + '\n')


class Logger:
    def __init__(self, log_dir):
        self.log_dir=log_dir
        self.data = OrderedDict()
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self,data, prefix='train',step=None,verbose=False):
        msg=f'{prefix} '
        for k, v in data.items():
            msg += f'{k} {v:.5f} '
            self.writer.add_scalar(f'{prefix}/{k}',v,step)

        if verbose:
            print(msg)
        with open(os.path.join(self.log_dir,f'{prefix}.txt'), 'a') as f:
            f.write(msg + '\n')

def print_shape(obj):
    if type(obj) == list or type(obj) == tuple:
        shapes = [item.shape for item in obj]
        print(shapes)
    else:
        print(obj.shape)

def overwrite_configs(cfg_base: dict, cfg: dict):
    keysNotinBase = []
    for key in cfg.keys():
        if key in cfg_base.keys():
            cfg_base[key] = cfg[key]
        else:
            keysNotinBase.append(key)
            cfg_base.update({key: cfg[key]})
    if len(keysNotinBase) != 0:
        print('==== WARNING: These keys are not set in DEFAULT_BASE_CONFIG... ====')
        print(keysNotinBase)
    return cfg_base

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

def dim_extend(data_list):
    results = []
    for i, tensor in enumerate(data_list):
        results.append(tensor[None,...])
    return results

class MultiGPUWrapper(nn.Module):
    def __init__(self,network,losses):
        super().__init__()
        self.network=network
        self.losses=losses

    def forward(self, data_gt):
        results={}
        data_pr=self.network(data_gt)
        results.update(data_pr)
        for loss in self.losses:
            results.update(loss(data_pr,data_gt,data_gt['step']))
        return results

class DummyLoss:
    def __init__(self,losses):
        self.keys=[]
        for loss in losses:
            self.keys+=loss.keys

    def __call__(self, data_pr, data_gt, step):
        return {key: data_pr[key] for key in self.keys}
