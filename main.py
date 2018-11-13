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
        # print(net_g)
        net_d, logits_real = SRGAN_d(t_target_image, is_test=False)
        # print(t_target_image)
        _, logits_fake = SRGAN_d(net_g, is_test=False)
        d_loss1 = fluid.layers.sigmoid_cross_entropy_with_logits(logits_real, 
                fluid.layers.ones(shape=logits_real.shape, dtype='float32'), 
                name='d1')
        d_loss2 = fluid.layers.sigmoid_cross_entropy_with_logits(logits_fake, 
                fluid.layers.ones(shape=logits_fake.shape, dtype='float32'), 
                name='d2')
        d_loss = d_loss1 + d_loss2
    with fluid.program_guard(g_program):
        # LR img
        t_image = fluid.layers.data(name='t_image_input_to_SRGAN_generator', shape=[-1, 3, 96, 96])
        # HR img
        t_target_image = fluid.layers.data(name='t_target_image', shape=[-1, 3, 384, 384])

        net_g = SRGAN_g(t_image, is_test=False)
        # net_d, logits_real = SRGAN_d(t_target_image, is_test=False)
        _, logits_fake = SRGAN_d(net_g, is_test=False)
        
        # 0.001 gan loss
        g_gan_loss = 1e-3 * fluid.layers.reduce_mean( 
                fluid.layers.sigmoid_cross_entropy_with_logits(logits_fake, 
                fluid.layers.ones(shape=logits_fake.shape,dtype='float32'), 
                name='g'))
        
        # mse loss
        mse_loss = fluid.layers.reduce_mean(fluid.layers.square_error_cost(net_g, t_target_image))

        # vgg loss
        ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        t_target_image_224 = fluid.layers.resize_bilinear(t_target_image, out_shape=[224, 224])  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
        t_predict_image_224 = fluid.layers.resize_bilinear(net_g, out_shape=[224, 224])  # resize_generate_image_for_vgg
        
        
        
        g_loss = mse_loss + g_gan_loss + vgg_loss

    
if __name__ == '__main__':
    train()
