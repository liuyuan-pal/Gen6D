import os

import torch
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.train_dataset import name2dataset
from network.loss import name2loss
from network import name2network
from train.lr_common_manager import name2lr_manager
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import dummy_collate_fn, simple_collate_fn


class Trainer:
    default_cfg={
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg":{
            "lr_init": 1.0e-4,
            "decay_step": 100000,
            "decay_rate": 0.5,
        },
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "worker_num": 8,
        'collate_fn': 'dummy',
        'train_loader_batch_size': 1,
        'val_loader_batch_size': 1,
    }
    def _init_dataset(self):
        self.train_set=name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)
        train_batch_size = self.cfg['train_loader_batch_size']
        collate_fn = dummy_collate_fn if self.cfg['collate_fn'] =='dummy' else simple_collate_fn
        self.train_set=DataLoader(self.train_set, train_batch_size, True, num_workers=self.cfg['worker_num'], collate_fn = collate_fn)
        print(f'train set len {len(self.train_set)}')
        self.val_set_list, self.val_set_names = [], []
        for val_set_cfg in self.cfg['val_set_list']:
            name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
            val_set = name2dataset[val_type](val_cfg, False)
            val_batch_size = self.cfg['val_loader_batch_size']
            val_set = DataLoader(val_set,val_batch_size,False,num_workers=self.cfg['worker_num'],collate_fn=collate_fn)
            self.val_set_list.append(val_set)
            self.val_set_names.append(name)
            print(f'{name} val set len {len(val_set)}')

    def _init_network(self):
        self.network=name2network[self.cfg['network']](self.cfg).cuda()

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu training
        if self.cfg['multi_gpus']:
            raise NotImplementedError
            # make multi gpu network
            # self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            # self.train_losses=[DummyLoss(self.val_losses)]
        else:
            self.train_network=self.network
            self.train_losses=self.val_losses

        if self.cfg['optimizer_type']=='adam':
            self.optimizer = Adam
        elif self.cfg['optimizer_type']=='sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator=ValidationEvaluator(self.cfg)
        self.lr_manager=name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])
        self.optimizer=self.lr_manager.construct_optimizer(self.optimizer,self.network)

    def __init__(self,cfg):
        self.cfg={**self.default_cfg,**cfg}
        self.model_name=cfg['name']
        self.model_dir=os.path.join('data/model',cfg['name'])
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')

    def run(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()

        best_para,start_step=self._load_model()
        train_iter=iter(self.train_set)

        pbar=tqdm(total=self.cfg['total_step'],bar_format='{r_bar}')
        pbar.update(start_step)
        for step in range(start_step,self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                # self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step']=step

            self.train_network.train()
            self.network.train()
            lr = self.lr_manager(self.optimizer, step)

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            log_info={}
            outputs=self.train_network(train_data)
            for loss in self.train_losses:
                loss_results = loss(outputs,train_data,step)
                for k,v in loss_results.items():
                    log_info[k]=v

            loss=0
            for k,v in log_info.items():
                if k.startswith('loss'):
                    loss=loss+torch.mean(v)

            loss.backward()
            self.optimizer.step()
            if ((step+1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info,step+1,'train')

            if (step+1)%self.cfg['val_interval']==0 or (step+1)==self.cfg['total_step']:
                torch.cuda.empty_cache()
                val_results, val_para = self.val_step(step)

                if val_para>best_para:
                    print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para=val_para
                    self._save_model(step+1,best_para,self.best_pth_fn)
                self._log_data(val_results,step+1,'val')

            if (step+1)%self.cfg['save_interval']==0:
                self._save_model(step+1,best_para)

            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),lr=lr)
            pbar.update(1)

        pbar.close()

    def val_step(self, step):
        val_results={}
        val_para = 0
        for vi, val_set in enumerate(self.val_set_list):
            val_results_cur, val_para_cur = self.val_evaluator(
                self.network, self.val_losses + self.val_metrics, val_set, step,
                self.model_name, val_set_name=self.val_set_names[vi])
            for k, v in val_results_cur.items():
                val_results[f'{self.val_set_names[vi]}-{k}'] = v
            # always use the final val set to select model!
            val_para = val_para_cur
        return val_results, val_para

    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.pth_fn):
            checkpoint=torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn=self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step':step,
            'best_para':best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self,results,step,prefix='train',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results,prefix,step,verbose)


class Analyzer(Trainer):
    def run(self):
        metric_names=['vis_loc_view', 'vis_bbox_scale']
        network_train_state=True
        dataset_train = True
        dataset_val_index = 0

        self._init_network()
        self._init_logger()

        best_para,start_step=self._load_model()


        if network_train_state:
            self.train_network.train()
            self.network.train()
        else:
            self.train_network.eval()
            self.network.eval()

        self.cfg['output_interval']=1
        val_metrics = [name2metrics[metric_name](self.cfg) for metric_name in metric_names]
        step = start_step

        if dataset_train:
            self.cfg['val_set_list']=[]
            self._init_dataset()
            train_iter = iter(self.train_set)
            for data_i in range(50):
                train_data = next(train_iter)
                with torch.no_grad():
                    train_data = to_cuda(train_data)
                # train_data['ref_imgs_info']['imgs'][:]=0
                outputs = self.train_network(train_data)
                for metric in val_metrics:
                    metric(outputs, train_data, step, data_index=data_i, model_name='train-'+self.model_name, output_root='data/analyze')
        else:
            val_set = self.val_set_list[dataset_val_index]
            for data_i, data in tqdm(range(len(val_set))):
                data = val_set[data_i]
                data = to_cuda(data)
                data['eval'] = True
                with torch.no_grad():
                    outputs = self.train_network(data)
                    for metric in val_metrics:
                        metric(outputs, data, step, data_index=data_i, model_name=self.val_set_names[dataset_val_index]+'-'+self.model_name, output_root='data/analyze')


