import time

import torch
import numpy as np
from tqdm import tqdm

from network.metrics import name2key_metrics
from train.train_tools import to_cuda


class ValidationEvaluator:
    def __init__(self,cfg):
        self.key_metric_name=cfg['key_metric_name']
        self.key_metric=name2key_metrics[self.key_metric_name]

    def __call__(self, model, losses, eval_dataset, step, model_name, val_set_name=None):
        if val_set_name is not None: model_name=f'{model_name}-{val_set_name}'
        model.eval()
        eval_results={}
        begin=time.time()
        for data_i, data in tqdm(enumerate(eval_dataset),total=len(eval_dataset),bar_format='{r_bar}'):
            data = to_cuda(data)
            data['eval']=True
            data['step']=step
            with torch.no_grad():
                outputs=model(data)
                for loss in losses:
                    loss_results=loss(outputs, data, step, data_index=data_i, model_name=model_name)
                    for k,v in loss_results.items():
                        if type(v)==torch.Tensor:
                            v=v.detach().cpu().numpy()

                        if k in eval_results:
                            eval_results[k].append(v)
                        else:
                            eval_results[k]=[v]

        for k,v in eval_results.items():
            eval_results[k]=np.concatenate(v,axis=0)

        key_metric_val=self.key_metric(eval_results)
        eval_results[self.key_metric_name]=key_metric_val
        print('eval cost {} s'.format(time.time()-begin))
        return eval_results, key_metric_val
