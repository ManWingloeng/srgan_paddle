import os
import re
import logging
from config import config, log_config
import numpy as np
from PIL import Image
from utils import *
def load_file_list(path=None, regx='.*.png', printable=False, keep_prefix=False):
    r"""Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : str or None
        A folder path, if `None`, use the current directory.
    regx : str
        The regx of file name.
    printable : boolean
        Whether to print the files infomation.
    keep_prefix : boolean
        Whether to keep path in the file name.

    Examples
    ----------
    >>> file_list = tl.files.load_file_list(path=None, regx='w1pre_[0-9]+\.(npz)')

    """
    if path is None:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    # return_list.sort()
    if keep_prefix:
        for i, f in enumerate(return_list):
            return_list[i] = os.path.join(path, f)

    if printable:
        logging.info('Match file list = %s' % return_list)
        logging.info('Number of files = %d' % len(return_list))
    return return_list

## preload DATA
train_hr_img_list = sorted(load_file_list(path=config.TRAIN.hr_img_path, keep_prefix=True))
# train_lr_img_list = sorted(load_file_list(path=config.TRAIN.lr_img_path, keep_prefix=True))
valid_hr_img_list = sorted(load_file_list(path=config.VALID.hr_img_path, keep_prefix=True))
# valid_lr_img_list = sorted(load_file_list(path=config.VALID.lr_img_path, keep_prefix=True))

def len_train_hr_img():
    return len(train_hr_img_list) # 800


def load_data(imgfile, shape):
    h, w = shape[1:]
    from PIL import Image
    im = Image.open(imgfile)

    # The storage order of the loaded image is W(widht),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them.
    im = im.resize((w, h), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))  # CHW
    im = im[(2, 1, 0), :, :]  # BGR

    # The mean to be subtracted from each image.
    # By default, the per-channel ImageNet mean.
    mean = np.array([104., 117., 124.], dtype=np.float32)
    mean = mean.reshape([3, 1, 1])
    im = im - mean
    return im.reshape([-1] + shape)


# print train_hr_img_list
def reader_creater_hr(list_file, root_dir, shuffle=True, return_name=False):
    # print(list_file.shape)
    def reader():
        while True:
            if shuffle:
                np.random.shuffle(list_file)
            for file in list_file:
                # file = file.strip("\n\r\t ")
                image = Image.open(file)
                image = np.array(image)
                image = crop_sub_imgs_fn(image)
                image_lr = downsample_fn(image)
                # print(image.shape)
                # image = image.resize((256, 256))
                # image = np.array(image) / 127.5 - 1
                # if len(image.shape) != 3:
                #     continue
                image = image[:, :, 0:3].astype("float32")
                image_lr = image_lr[:, :, 0:3].astype("float32")
                image = image.transpose([2, 0, 1])
                image_lr = image_lr.transpose([2, 0, 1])
                # print(image[np.newaxis, :].shape)
                if return_name:
                    yield image[np.newaxis, :], image_lr[np.newaxis, :], os.path.basename(file)
                else:
                    yield image[np.newaxis, :], image_lr[np.newaxis, :]

    return reader

def train_hr_reader():
    return reader_creater_hr(train_hr_img_list, root_dir=config.TRAIN.hr_img_path)

# We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r=4
# def train_lr_reader():
    # return reader_creater_lr(train_lr_img_list, root_dir=config.TRAIN.lr_img_path, cycle=False, return_name=True)

def val_hr_reader():
    return reader_creater_hr(valid_hr_img_list, root_dir=config.VALID.hr_img_path)

# def val_lr_reader():
    # return reader_creater(valid_lr_img_list, root_dir=config.VALID.lr_img_path, cycle=False, return_name=True)












## threading to preprocess the data
def threading_data(data=None, fn=None, thread_count=None, **kwargs):
    """Process a batch of data by given function by threading.

    Usually be used for data augmentation.

    Parameters
    -----------
    data : numpy.array or others
        The data to be processed.
    thread_count : int
        The number of threads to use.
    fn : function
        The function for data processing.
    more args : the args for `fn`
        Ssee Examples below.

    Examples
    --------
    Process images.

    >>> images, _, _, _ = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))
    >>> images = tl.prepro.threading_data(images[0:32], tl.prepro.zoom, zoom_range=[0.5, 1])

    Customized image preprocessing function.

    >>> def distort_img(x):
    >>>     x = tl.prepro.flip_axis(x, axis=0, is_random=True)
    >>>     x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    >>>     x = tl.prepro.crop(x, 100, 100, is_random=True)
    >>>     return x
    >>> images = tl.prepro.threading_data(images, distort_img)

    Process images and masks together (Usually be used for image segmentation).

    >>> X, Y --> [batch_size, row, col, 1]
    >>> data = tl.prepro.threading_data([_ for _ in zip(X, Y)], tl.prepro.zoom_multi, zoom_range=[0.5, 1], is_random=True)
    data --> [batch_size, 2, row, col, 1]
    >>> X_, Y_ = data.transpose((1,0,2,3,4))
    X_, Y_ --> [batch_size, row, col, 1]
    >>> tl.vis.save_image(X_, 'images.png')
    >>> tl.vis.save_image(Y_, 'masks.png')

    Process images and masks together by using ``thread_count``.

    >>> X, Y --> [batch_size, row, col, 1]
    >>> data = tl.prepro.threading_data(X, tl.prepro.zoom_multi, 8, zoom_range=[0.5, 1], is_random=True)
    data --> [batch_size, 2, row, col, 1]
    >>> X_, Y_ = data.transpose((1,0,2,3,4))
    X_, Y_ --> [batch_size, row, col, 1]
    >>> tl.vis.save_image(X_, 'after.png')
    >>> tl.vis.save_image(Y_, 'before.png')

    Customized function for processing images and masks together.

    >>> def distort_img(data):
    >>>    x, y = data
    >>>    x, y = tl.prepro.flip_axis_multi([x, y], axis=0, is_random=True)
    >>>    x, y = tl.prepro.flip_axis_multi([x, y], axis=1, is_random=True)
    >>>    x, y = tl.prepro.crop_multi([x, y], 100, 100, is_random=True)
    >>>    return x, y

    >>> X, Y --> [batch_size, row, col, channel]
    >>> data = tl.prepro.threading_data([_ for _ in zip(X, Y)], distort_img)
    >>> X_, Y_ = data.transpose((1,0,2,3,4))

    Returns
    -------
    list or numpyarray
        The processed results.

    References
    ----------
    - `python queue <https://pymotw.com/2/Queue/index.html#module-Queue>`__
    - `run with limited queue <http://effbot.org/librarybook/queue.htm>`__

    """
    import threading
    def apply_fn(results, i, data, kwargs):
        results[i] = fn(data, **kwargs)

    if thread_count is None:
        results = [None] * len(data)
        threads = []
        # for i in range(len(data)):
        #     t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, data[i], kwargs))
        for i, d in enumerate(data):
            
            t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, d, kwargs))
            t.start()
            threads.append(t)
    else:
        divs = np.linspace(0, len(data), thread_count + 1)
        divs = np.round(divs).astype(int)
        results = [None] * thread_count
        threads = []
        for i in range(thread_count):
            t = threading.Thread(
                name='threading_and_return', target=apply_fn, args=(results, i, data[divs[i]:divs[i + 1]], kwargs)
            )
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    if thread_count is None:
        try:
            return np.asarray(results)
        except Exception:
            return results
    else:
        return np.concatenate(results)



# import paddle.v2 as paddle
# import paddle.fluid as fluid
# # HR_train_reader = train_hr_reader()
# # data_thr, data_tlr=HR_train_reader.next()
# THR_reader = paddle.batch(train_hr_reader(), 16)()
# # print(next(THR_reader))
# data=next(THR_reader)
# data_thr=[]
# data_tlr=[]
# for thr,tlr in data:
#     data_thr.append(thr)
#     data_tlr.append(tlr)
# data_thr=np.array(data_thr)
# data_thr=np.squeeze(data_thr)

# data_tlr=np.array(data_tlr)
# data_tlr=np.squeeze(data_tlr)
# print("data_thr:",data_thr.shape)
# print("data_tlr:",data_tlr.shape)
# # for img in data_thr:
#     # print img
# # for img in data_thr:
# #     img=np.squeeze(img)
# #     print(img.shape)
# # data_tlr=[downsample_fn(np.squeeze(img)) for img in data_thr]
# # print(data_tlr)
# # 
# print(data_tlr.shape)
# import paddle.v2 as paddle
# import paddle.fluid as fluid

# infer_program = fluid.default_main_program().clone(for_test=True)
# with fluid.program_guard(infer_program):
#     data = fluid.layers.data(name='data',shape=[-1, 3, 384, 384])
#     # data_lr = fluid.layers.image_resize(input=data, out_shape=[96, 96], resample='')
#     data_lr = fluid.layers.data(name='data_lr', shape=[-1, 3, 96, 96])
#     predict = fluid.layers.fc(input=data, size=1, act='relu')
#     softmax = fluid.layers.softmax(predict)
#     print(data.shape)

# place = fluid.CPUPlace()
# exe = fluid.Executor(place)
# exe.run(fluid.default_startup_program())
# p = exe.run(infer_program, fetch_list=[predict, softmax], feed={
#     'data': data_thr,
#     'data_lr': data_tlr
# })
# print(p)
# data_tlr=threading_data(data_thr, fn=downsample_fn)

# print(data_tlr.shape)