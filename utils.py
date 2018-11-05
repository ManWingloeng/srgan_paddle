# import tensorflow as tf
# import tensorlayer as tl
# from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import paddle
import paddle.fluid as fluid

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = fluid.layers.random_crop(x, shape=[3, 384, 384])
    # x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x

def imresize(x, size=None, interp='bicubic', mode=None):
    """Resize an image by given output size and method.

    Warning, this function will rescale the value to [0, 255].

    Parameters
    -----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    size : list of 2 int or None
        For height and width.
    interp : str
        Interpolation method for re-sizing (`nearest`, `lanczos`, `bilinear`, `bicubic` (default) or `cubic`).
    mode : str
        The PIL image mode (`P`, `L`, etc.) to convert arr before resizing.

    Returns
    -------
    numpy.array
        A processed image.

    References
    ------------
    - `scipy.misc.imresize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html>`__

    """
    if size is None:
        size = [100, 100]
    # channel loc is wrong in paddle????????
    # if x.shape[-1] == 1:
    #     # greyscale
    #     x = scipy.misc.imresize(x[:, :, 0], size, interp=interp, mode=mode)
    #     return x[:, :, np.newaxis]
    # elif x.shape[-1] == 3:
    #     # rgb, bgr ..
    #     return scipy.misc.imresize(x, size, interp=interp, mode=mode)
    # else:
    #     raise Exception("Unsupported channel %d" % x.shape[-1])
    if x.shape[1] == 1:
        # greyscale
        x = scipy.misc.imresize(x[0, :, :], size, interp=interp, mode=mode)
        return x[:, :, np.newaxis]
    elif x.shape[1] == 3:
        # rgb, bgr ..
        return scipy.misc.imresize(x, size, interp=interp, mode=mode)
    else:
        raise Exception("Unsupported channel %d" % x.shape[1])

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x
