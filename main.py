# -*-coding:utf-8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import paddle
import paddle.fluid as fluid
from model import SRGAN_g, SRGAN_d
import vgg
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

        
        



    ###========================== DEFINE MODEL ============================###
    ## train inference
     
    d_program = fluid.Program()
    g_program = fluid.Program()
    with fluid.program_guard(d_program):
        # LR img
        t_image = fluid.layers.data(name='t_image_input_to_SRGAN_generator', shape=[-1, 3, 96, 96])
        # HR img
        t_target_image = fluid.layers.data(name='t_target_image', shape=[-1, 3, 384, 384])

        net_g = SRGAN_g(t_image, is_test=False)
        print(net_g)
        net_d, logits_real = SRGAN_d(t_target_image, is_test=False)
        print(t_target_image)
        _, logits_fake = SRGAN_d(net_g, is_test=False)
    
if __name__ == '__main__':
    train()
