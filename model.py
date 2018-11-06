import time
import paddle
import paddle.fluid as fluid
from layers import *

def SRGAN_g(t_image, is_test=False):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    with fluid.unique_name.guard("SRGAN_g"):
        n = t_image
        n = conv(n, 64, 3, act='relu', name='n64s1/c')
        temp = n
        for i in range(16):
            nn = conv(n, 64, 3, name='n64s1/c1/%s' % i)
            nn = bn_relu(nn, is_test=is_test, name='n64s1/b1/%s' % i)
            nn = conv(nn, 64, 3, name='n64s1/c2/%s' % i)
            nn = bn_relu(nn, is_test=is_test, name='n64s1/b2/%s' % i)
            nn = elementwise_add(n, nn, name='b_residual_add/%s' % i)
            n = nn