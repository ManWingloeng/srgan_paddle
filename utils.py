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

def crop(x, wrg, hrg, is_random=False, row_index=0, col_index=1):
    """Randomly or centrally crop an image.

    Parameters
    ----------
    x : numpy.array
        An image with dimension of [row, col, channel] (default).
    wrg : int
        Size of width.
    hrg : int
        Size of height.
    is_random : boolean,
        If True, randomly crop, else central crop. Default is False.
    row_index: int
        index of row.
    col_index: int
        index of column.

    Returns
    -------
    numpy.array
        A processed image.

    """
    h, w = x.shape[row_index], x.shape[col_index]

    if (h <= hrg) or (w <= wrg):
        raise AssertionError("The size of cropping should smaller than the original image")

    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
        # tl.logging.info(h_offset, w_offset, x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset].shape)
        return x[h_offset:hrg + h_offset, w_offset:wrg + w_offset]
    else:  # central crop
        h_offset = int(np.floor((h - hrg) / 2.))
        w_offset = int(np.floor((w - wrg) / 2.))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return x[h_offset:h_end, w_offset:w_end]

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
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
    if x.shape[0] == 1:
        # greyscale
        x = scipy.misc.imresize(x[0, :, :], size, interp=interp, mode=mode)
        return x[:, :, np.newaxis]
    elif x.shape[0] == 3:
        # rgb, bgr ..
        return scipy.misc.imresize(x, size, interp=interp, mode=mode)
    else:
        raise Exception("Unsupported channel %d" % x.shape[0])

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = scipy.misc.imresize(x, size=[96, 96], interp='bicubic', mode=None)
    # x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    # shapex=x.shape
    # print("shapex",shapex)
    # x = (x - 0.5)*2
    return x
