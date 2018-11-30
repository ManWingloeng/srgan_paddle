import time
import paddle.v2 as paddle
import paddle.fluid as fluid
from layers import *

def SRGAN_g(t_image, is_test=False, name='G'):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    with fluid.unique_name.guard(name+'_'):
        n = t_image
        n = conv(n, 64, 3, act='relu', name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = conv(n, 64, 3, name='n64s1/c1/%s' % i)
            nn = bn(nn, act='relu', is_test=is_test, name='n64s1/b1/%s' % i)
            nn = conv(nn, 64, 3, name='n64s1/c2/%s' % i)
            nn = bn(nn, act='relu', is_test=is_test, name='n64s1/b2/%s' % i)
            nn = elementwise_add(n, nn, name='b_residual_add/%s' % i)
            n = nn
        
        n = conv(n, 64, 3, name='n64s1/c/m')
        n = bn(n, name='n64s1/b/m')
        n = elementwise_add(n, temp, name='add3')
        # B residual blacks end

        n = conv(n, 256, 3, name='n256s1/1')
        n = SubpixelConv_relu(n, scale=2, name='pixelshufflerx2/1')

        n = conv(n, 256, 3, name='n256s1/2')
        n = SubpixelConv_relu(n, scale=2, name='pixelshufflerx2/2')

        n = conv(n, 3, 1, act='tanh', name='out')
        return n

def SRGAN_g2(t_image, is_test=False, name='G2'):
    size = (t_image.shape).as_list()
    with fluid.unique_name.guard():
        n = t_image
        n = conv(n, 64, 3, act='relu', name='n64s1/c')
        temp = n
        
        # B residual blocks
        for i in range(16):
            nn = conv(n, 64, 3, name='n64s1/c1/%s' % i)
            nn = bn(nn, act='relu', is_test=is_test, name='n64s1/b1/%s' % i)
            nn = conv(nn, 64, 3, name='n64s1/c2/%s' % i)
            nn = bn(nn, act='relu', is_test=is_test, name='n64s1/b2/%s' % i)
            nn = elementwise_add(n, nn, name='b_residual_add/%s' % i)
            n = nn

        n = conv(n, 64, 3, name='n64s1/c/m')
        n = bn(n, is_test=is_test, name='n64s1/b/m')
        n = elementwise_add(n, temp, name='add3')
        # B residual blacks end

        n = UpSampling2dLayer(n, out_shape=[size[2] * 2, size[3] * 2], method='NEAREST', name='up1/upsample2d')
        n = conv(n, 64, 3, name='up1/conv2d')
        n = bn(n, act='relu', is_test=is_test, name='up1/batch_norm')

        n = UpSampling2dLayer(n, out_shape=[size[2] * 4, size[3] * 4], method='NEAREST', name='up2/upsample2d')
        n = conv(n, 32, 3, name='up2/conv2d')
        n = bn(n, act='relu', is_test=is_test, name='up2/batch_norm')

        n = conv(n, 3, 1, act='relu', name='out')
        return n

def SRGAN_d2(t_image, is_test=False, name='D2'):
    with fluid.unique_name.guard():
        n = t_image
        n = conv(n, 63, 3, act='leaky_relu', name='n64s1/c')

        n = conv(n, 64, 3, (2, 2), act='leaky_relu', name='n64s2/c')
        n = bn(n, is_test=is_test, name='n64s2/b')

        n = conv(n, 128, 3, act='leaky_relu', name='n128s1/c')
        n = bn(n, is_test=is_test, name='n128s1/b')

        n = conv(n, 128, 3, (2, 2), act='leaky_relu', name='n128s2/c')
        n = bn(n, is_test=is_test, name='n128s2/b')

        n = conv(n, 256, 3, act='leaky_relu', name='n256s1/c')
        n = bn(n, is_test=is_test, name='n256s1/b')

        n = conv(n, 256, 3, (2, 2), act='leaky_relu', name='n256s2/c')
        n = bn(n, is_test=is_test, name='n256s2/b')

        n = conv(n, 512, 3, act='leaky_relu', name='n512s1/c')
        n = bn(n, is_test=is_test, name='n512s1/b')

        n = conv(n, 512, 3, (2, 2), act='leaky_relu', name='n512s2/c')
        n = bn(n, is_test=is_test, name='n512s2/b')

        n = fluid.layers.flatten(n, name='flatten')
        n = fully_connected(n, units=1024, act='leaky_relu', name='fc1024')
        n = fully_connected(n, units=1, name='out')

        logits = n
        n = fluid.layers.sigmoid(n)

        return n, logits

def SRGAN_d(input_images, is_test=False, name='D'):
    df_dim = 64
    with fluid.unique_name.guard():
        net_in = input_images
        net_h0 = conv(net_in, df_dim , 4, (2, 2), act='leaky_relu', name='h0/c')

        net_h1 = conv(net_h0, df_dim*2, 4, (2, 2), name='h1/c')
        net_h1 = bn(net_h1, act='leaky_relu', is_test=is_test, name='h1/bn')
        net_h2 = conv(net_h1, df_dim * 4, 4, (2, 2), name='h2/c')
        net_h2 = bn(net_h2, act='leaky_relu', is_test=is_test,  name='h2/bn')
        net_h3 = conv(net_h2, df_dim * 8, 4, (2, 2), name='h3/c')
        net_h3 = bn(net_h3, act='leaky_relu', is_test=is_test, name='h3/bn')
        net_h4 = conv(net_h3, df_dim * 16, 4, (2, 2), name='h4/c')
        net_h4 = bn(net_h4, act='leaky_relu', is_test=is_test,  name='h4/bn')
        net_h5 = conv(net_h4, df_dim * 32, 4, (2, 2), name='h5/c')
        net_h5 = bn(net_h5, act='leaky_relu', is_test=is_test, name='h5/bn')
        net_h6 = conv(net_h5, df_dim * 16, 1, (1, 1), name='h6/c')
        net_h6 = bn(net_h6, act='leaky_relu', is_test=is_test, name='h6/bn')
        net_h7 = conv(net_h6, df_dim * 8, 1, (1, 1), name='h7/c')
        net_h7 = bn(net_h7, is_test=is_test, name='h7/bn')

        net = conv(net_h7, df_dim * 2, 1, (1, 1), name='res/c')
        net = bn(net, act='leaky_relu', is_test=is_test,  name='res/bn')
        net = conv(net, df_dim * 2, 3, (1, 1), name='res/c2')
        net = bn(net, act='leaky_relu', is_test=is_test, name='res/bn2')
        net = conv(net, df_dim * 8, 3, (1, 1), name='res/c3')
        net = bn(net, is_test=is_test, name='res/bn3')
        net_h8 = elementwise_add(net_h7, net, name='res/add')
        net_h8 = fluid.layers.sigmoid(net_h8)

        net_ho = fluid.layers.flatten(net_h8, name='h0/flatten')
        net_ho = fully_connected(net_h0, units=1, name='h0/fc')
        # net_ho = fluid.layers.identity()
        logits = net_ho
        net_ho = fluid.layers.sigmoid(net_ho)

        return net_ho, logits