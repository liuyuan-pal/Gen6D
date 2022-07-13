import abc

class LearningRateManager(abc.ABC):
    @staticmethod
    def set_lr_for_all(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def construct_optimizer(self, optimizer, network):
        # may specify different lr for different parts
        # use group to set learning rate
        paras = network.parameters()
        return optimizer(paras, lr=1e-3)

    @abc.abstractmethod
    def __call__(self, optimizer, step, *args, **kwargs):
        pass

class ExpDecayLR(LearningRateManager):
    def __init__(self,cfg):
        self.lr_init=cfg['lr_init']
        self.decay_step=cfg['decay_step']
        self.decay_rate=cfg['decay_rate']
        self.lr_min=1e-5

    def __call__(self, optimizer, step, *args, **kwargs):
        lr=max(self.lr_init*(self.decay_rate**(step//self.decay_step)),self.lr_min)
        self.set_lr_for_all(optimizer,lr)
        return lr

class WarmUpExpDecayLR(LearningRateManager):
    def __init__(self, cfg):
        self.lr_warm=cfg['lr_warm']
        self.warm_step=cfg['warm_step']
        self.lr_init=cfg['lr_init']
        self.decay_step=cfg['decay_step']
        self.decay_rate=cfg['decay_rate']
        self.lr_min=1e-5

    def __call__(self, optimizer, step, *args, **kwargs):
        if step<self.warm_step:
            lr=self.lr_warm
        else:
            lr=max(self.lr_init*(self.decay_rate**((step-self.warm_step)//self.decay_step)),self.lr_min)
        self.set_lr_for_all(optimizer,lr)
        return lr

name2lr_manager={
    'exp_decay': ExpDecayLR,
    'warm_up_exp_decay': WarmUpExpDecayLR,
}