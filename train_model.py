import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/train/')
flags = parser.parse_args()

trainer = Trainer(load_cfg(flags.cfg),)
trainer.run()