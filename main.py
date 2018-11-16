# -*-coding:utf-8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import paddle
import paddle.fluid as fluid
from model import SRGAN_g, SRGAN_d
# import vgg
from utils import *
from config import config, log_config

# we improt VGG19
from vgg19 import VGG19


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

def get_param(program, prefix):
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.startswith(prefix)]


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

        # Generate the HR img from LR
        net_g = SRGAN_g(t_image, is_test=False)
        
        ## clone for test
        net_g_test_program = g_program.clone(for_test=True)
        
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
        
        # vgg19_program, vgg19_feed_names, vgg19_fetch_targets = fluid.io.load_inference_model('./VGG19_pd_model_param', 
        #                                                            exe, 'vgg19_model', 'vgg19_params')
        
        ### VGG19 is not resize the input!!!need implement!!! ###
        vgg_target_emb = VGG19().net((t_target_image_224 + 1) / 2)
        vgg_predict_emb = VGG19().net((t_predict_image_224 + 1) / 2)


        vgg_loss = 2e-6 * fluid.layers.reduce_mean(fluid.layers.square_error_cost(
            vgg_predict_emb, vgg_target_emb))

        g_loss = mse_loss + g_gan_loss + vgg_loss

    g_vars = get_param(g_program, prefix='G')
    d_vars = get_param(d_program, prefix='D')

    opt = fluid.optimizer.Adam(learning_rate=lr_init, beta1=beta1)

    ## pretrain
    opt.minimize(loss=mse_loss, parameter_list=g_vars)
    ## SRGAN
    opt.minimize(loss=d_loss, parameter_list=d_vars)
    opt.minimize(loss=g_loss, parameter_list=g_vars)

    
if __name__ == '__main__':
    train()
