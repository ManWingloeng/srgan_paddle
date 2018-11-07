# -*-coding:utf-8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import paddle
import paddle.fluid as fluid
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def train():
        ## create folders to save result images and trained model
        save_dir_ginit = "samples/train_ginit"
        save_dir_gan = "samples/train_gan"
        if not os.path.isdir(save_dir_ginit):
            os.makedirs(save_dir_ginit)
        if not os.path.isdir(save_dir_gan):
            os.makedirs(save_dir_gan)
        checkpoint_dir = "checkpoint"
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
