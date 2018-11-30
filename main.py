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
import data_reader

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
    g_pretrain_program = fluid.Program()


    opt = fluid.optimizer.Adam(learning_rate=lr_init, beta1=beta1)
    # with fluid.program_guard(g_pretrain_program):
    #     # # LR img
    #     t_image = fluid.layers.data(name='t_image', shape=[3, 96, 96])
    #     # HR img
    #     t_target_image = fluid.layers.data(name='t_target_image', shape=[3, 384, 384])


    #     # Generate the HR img from LR
    #     net_g = SRGAN_g(t_image, is_test=False)
    #     # mse loss
    #     mse_loss = fluid.layers.reduce_mean(fluid.layers.square_error_cost(net_g, t_target_image))
    #     ## pretrain
    #     g_vars = get_param(g_program, prefix='G')
    #     opt.minimize(loss=mse_loss, parameter_list=g_vars)
    with fluid.program_guard(d_program):
        # LR img
        t_image = fluid.layers.data(name='t_lr_image', shape=[3, 96, 96])
    #     print("t_image numel:",t_image.numel)
        # HR img
        t_target_image = fluid.layers.data(name='t_hr_image', shape=[3, 384, 384])
    #     print("t_target_image numel:",t_target_image.numel)
        net_g = SRGAN_g(t_image, is_test=False)
        print("net_g:",net_g)
        net_d, logits_real = SRGAN_d(t_target_image, is_test=False)
        # print(t_target_image)
        _, logits_fake = SRGAN_d(net_g, is_test=False)

        ones_real = fluid.layers.fill_constant_batch_size_like(logits_real, shape=[-1, 1], dtype='float32', value=1)
        zeros_fake = fluid.layers.fill_constant_batch_size_like(logits_fake, shape=[-1, 1], dtype='float32', value=0)

        d_loss1 = fluid.layers.reduce_mean(
            fluid.layers.sigmoid_cross_entropy_with_logits(x=logits_real, 
                label=ones_real, 
                name='d1'))
        d_loss2 = fluid.layers.reduce_mean(
            fluid.layers.sigmoid_cross_entropy_with_logits(x=logits_fake, 
                label=zeros_fake, 
                name='d2'))
        d_loss = d_loss1 + d_loss2

    with fluid.program_guard(g_program):
        # LR img
        t_image = fluid.layers.data(name='t_lr_image', shape=[3, 96, 96])
        # HR img
        t_target_image = fluid.layers.data(name='t_hr_image', shape=[3, 384, 384])

        # Generate the HR img from LR
        net_g = SRGAN_g(t_image, is_test=False)
        print("net_g:",net_g)
        ## clone for test
        net_g_test_program = g_program.clone(for_test=True)

        # net_d, logits_real = SRGAN_d(t_target_image, is_test=False)
        _, logits_fake = SRGAN_d(net_g, is_test=False)
        
        ones_fake = fluid.layers.fill_constant_batch_size_like(logits_fake, shape=[-1, 1], dtype='float32', value=1)
        # 0.001 gan loss
        g_gan_loss = 1e-3 * fluid.layers.reduce_mean( 
                fluid.layers.sigmoid_cross_entropy_with_logits(logits_fake, 
                ones_fake, 
                name='g'))

        # mse loss
        mse_loss = fluid.layers.reduce_mean(fluid.layers.square_error_cost(net_g, t_target_image))

        # vgg loss
        ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        t_target_image_224 = fluid.layers.resize_bilinear(t_target_image, out_shape=[224, 224])  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
        t_predict_image_224 = fluid.layers.resize_bilinear(net_g, out_shape=[224, 224])  # resize_generate_image_for_vgg
        ## maybe just data layers is ok, preprocess before feed
        # t_target_image_224 = fluid.layers.data(name='t_target_image_224', shape=[3, 224, 224])
        # t_predict_image_224 = fluid.layers.data(name='t_predict_image_224', shape=[3, 224, 224])



        # vgg19_program, vgg19_feed_names, vgg19_fetch_targets = fluid.io.load_inference_model('./VGG19_pd_model_param', 
        #                                                            exe, 'vgg19_model', 'vgg19_params')

        # print t_target_image_224.shape
        # tt_input = (t_target_image_224 + 1) / 2
        # print tt_input
        # vgg_target_emb = VGG19().net(t_target_image_224)
        # vgg_predict_emb = VGG19().net(t_predict_image_224)
        vgg_target_emb = VGG19().net((t_target_image_224 + 1) / 2)
        vgg_predict_emb = VGG19().net((t_predict_image_224 + 1) / 2)

        vgg_loss = 2e-6 * fluid.layers.reduce_mean(fluid.layers.square_error_cost(
            vgg_predict_emb, vgg_target_emb))

        g_loss = mse_loss + g_gan_loss + vgg_loss

    g_vars = get_param(g_program, prefix='G')
    d_vars = get_param(d_program, prefix='D')
    # print("g_vars:\n",g_vars)
    ## SRGAN
    opt.minimize(loss=g_loss, parameter_list=g_vars)
    opt.minimize(loss=d_loss, parameter_list=d_vars)

    #     place = fluid.CUDAPlace(1) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        ## VGG19 load params
        # fluid.io.load_params(exe, "/home/aistudio/work/data/vgg_pd_params")
        fluid.io.load_params(exe, "/home/aistudio/work/data/vgg_pd_params")

        ## reader
        #     batch_train_hr_reader = paddle.batch(data_reader.train_hr_reader(), batch_size)()
        #     max_imgs = data_reader.len_train_hr_img()
        from data_reader import *
        batch_train_hr_reader = paddle.batch(train_hr_reader(), batch_size)()
        max_imgs = len_train_hr_img()
        for epoch in range(0, n_epoch_init + 1):
            epoch_time = time.time()
            epoch_d_fake_loss = []
            epoch_d_real_loss = []
            epoch_d_loss = []
            epoch_g_gan_loss = []
            epoch_g_mse_loss = []
            epoch_g_vgg_loss = []
            epoch_g_loss = []
            epoch_mse_loss = []
            total_mse_loss, batch_id = 0, 0
            for idx in range(0, max_imgs, batch_size):
                print("Training Epoch {} batch {}").format(epoch, idx)
                data=next(batch_train_hr_reader)
        #             data_thr=[]
        #             data_tlr=[]
        #             for thr,tlr in data:
        #                 data_thr.append(thr)
        #                 data_tlr.append(tlr)
                data_thr = map(lambda x: x[0], data)
                data_tlr = map(lambda x: x[1], data)

                data_thr=np.array(data_thr)
                data_thr=np.squeeze(data_thr)

                data_tlr=np.array(data_tlr)
                data_tlr=np.squeeze(data_tlr)
    #             print("data_thr",data_thr.shape)
    #             print("data_tlr",data_tlr.shape)
        #         print(data_thr.numel)
        #         print(data)
        #         _mse_loss = exe.run(program=g_pretrain_program, fetch_list=[mse_loss], feed={
        #             't_image':data_tlr,
        #             't_target_image':data_thr
        #         })
        #         epoch_mse_loss.append(np.mean(_mse_loss))

                _d_loss,_d_loss1,_d_loss2 = exe.run(program=d_program, fetch_list=[d_loss,d_loss1,d_loss2],feed={
                    't_lr_image':data_tlr,
                    't_hr_image':data_thr,              
                })
                epoch_d_fake_loss.append(_d_loss2)
                epoch_d_real_loss.append(_d_loss1)
                epoch_d_loss.append(_d_loss)


                _g_loss, _g_mse_loss, _vgg_loss, _g_gan_loss = exe.run(program=g_program, fetch_list=[g_loss, mse_loss, vgg_loss, g_gan_loss],feed={
                    't_lr_image':data_tlr,
                    't_hr_image':data_thr,                
                })
                epoch_g_gan_loss.append(_g_gan_loss)
                epoch_g_mse_loss.append(_g_mse_loss)
                epoch_g_vgg_loss.append(_vgg_loss)
                epoch_g_loss.append(_g_loss)
                if idx % 50 == 0:
                    print("Epoch {} batch {}:\n d_loss {} | d_fake_loss {} | d_real_loss {}\n g_loss {} | g_gan_loss {} | g_mse_loss {} | g_vgg_loss {}\n ".format(
                        epoch, idx, np.mean(epoch_d_loss), np.mean(epoch_d_fake_loss), 
                        np.mean(epoch_d_real_loss), np.mean(epoch_g_loss), np.mean(epoch_g_gan_loss), 
                        np.mean(epoch_g_mse_loss), np.mean(epoch_g_vgg_loss)))


    
if __name__ == '__main__':
    train()
