# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

import numpy as np
import paddle.v2 as paddle
import paddle.fluid as fluid
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float("bn_decay", 0.9, "batch norm decay")
flags.DEFINE_float("dropout_rate", 0.5, "dropout rate")

flag_weight_decay = 0.0004
def calc_padding(img_width, stride, dilation, filter_width):
    """ calculate pixels to padding in order to keep input/output size same. """

    filter_width = dilation * (filter_width - 1) + 1
    if img_width % stride == 0:
        pad_along_width = max(filter_width - stride, 0)
    else:
        pad_along_width = max(filter_width - (img_width % stride), 0)
    return pad_along_width // 2, pad_along_width - pad_along_width // 2

def elementwise_add(x, y, axis=-1, act=None, name=None):
    return fluid.layers.elementwise_add(x, y, axis=-1, act=None, name=None)

def conv(inputs,
         filters,
         kernel,
         strides=(1, 1),
         dilation=(1, 1),
         act = None,
         num_groups=1,
        #  w_init=True,
         name=None,
         conv_param=None):
    """ normal conv layer """

    if isinstance(kernel, (tuple, list)):
        n = operator.mul(*kernel) * inputs.shape[1]
    else:
        n = kernel * kernel * inputs.shape[1]

    # if w_init:
    #     w_param = fluid.initializer.NormalInitializer
    # pad input
    if isinstance(kernel, int):
        kernel=(kernel, kernel)
    padding = (0, 0, 0, 0) \
        + calc_padding(inputs.shape[2], strides[0], dilation[0], kernel[0]) \
        + calc_padding(inputs.shape[3], strides[1], dilation[1], kernel[1])
    if sum(padding) > 0:
        inputs = fluid.layers.pad(inputs, padding, 0)

    param_attr = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.NormalInitializer(
            0.0, scale=np.sqrt(2.0 / n)),
        regularizer=fluid.regularizer.L2Decay(flag_weight_decay))

    bias_attr = fluid.param_attr.ParamAttr(
        regularizer=fluid.regularizer.L2Decay(0.))

    return fluid.layers.conv2d(
        inputs,
        filters,
        kernel,
        stride=strides,
        padding=0,
        dilation=dilation,
        groups=num_groups,
        name=name,
        param_attr=param_attr if conv_param is None else conv_param,
        use_cudnn=False if num_groups == inputs.shape[1] == filters else True,
        bias_attr=bias_attr,
        act=act)


def sep(inputs, filters, kernel, strides=(1, 1), dilation=(1, 1)):
    """ Separable convolution layer """

    if isinstance(kernel, (tuple, list)):
        n_depth = operator.mul(*kernel)
    else:
        n_depth = kernel * kernel
    n_point = inputs.shape[1]

    if isinstance(strides, (tuple, list)):
        multiplier = strides[0]
    else:
        multiplier = strides

    depthwise_param = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.NormalInitializer(
            0.0, scale=np.sqrt(2.0 / n_depth)),
        regularizer=fluid.regularizer.L2Decay(flag_weight_decay))

    pointwise_param = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.NormalInitializer(
            0.0, scale=np.sqrt(2.0 / n_point)),
        regularizer=fluid.regularizer.L2Decay(flag_weight_decay))

    depthwise_conv = conv(
        inputs=inputs,
        kernel=kernel,
        filters=int(filters * multiplier),
        strides=strides,
        dilation=dilation,
        num_groups=int(filters * multiplier),
        conv_param=depthwise_param)

    return conv(
        inputs=depthwise_conv,
        kernel=(1, 1),
        filters=int(filters * multiplier),
        strides=(1, 1),
        dilation=dilation,
        conv_param=pointwise_param)


def maxpool(inputs, kernel, strides=(1, 1)):
    padding = (0, 0, 0, 0) \
              + calc_padding(inputs.shape[2], strides[0], 1, kernel[0]) \
              + calc_padding(inputs.shape[3], strides[1], 1, kernel[1])
    if sum(padding) > 0:
        inputs = fluid.layers.pad(inputs, padding, 0)

    return fluid.layers.pool2d(
        inputs, kernel, 'max', strides, pool_padding=0, ceil_mode=False)


def avgpool(inputs, kernel, strides=(1, 1)):
    padding_pixel = (0, 0, 0, 0)
    padding_pixel += calc_padding(inputs.shape[2], strides[0], 1, kernel[0])
    padding_pixel += calc_padding(inputs.shape[3], strides[1], 1, kernel[1])

    if padding_pixel[4] == padding_pixel[5] and padding_pixel[
            6] == padding_pixel[7]:
        # same padding pixel num on all sides.
        return fluid.layers.pool2d(
            inputs,
            kernel,
            'avg',
            strides,
            pool_padding=(padding_pixel[4], padding_pixel[6]),
            ceil_mode=False)
    elif padding_pixel[4] + 1 == padding_pixel[5] and padding_pixel[6] + 1 == padding_pixel[7] \
            and strides == (1, 1):
        # different padding size: first pad then crop.
        x = fluid.layers.pool2d(
            inputs,
            kernel,
            'avg',
            strides,
            pool_padding=(padding_pixel[5], padding_pixel[7]),
            ceil_mode=False)
        x_shape = x.shape
        return fluid.layers.crop(
            x,
            shape=(-1, x_shape[1], x_shape[2] - 1, x_shape[3] - 1),
            offsets=(0, 0, 1, 1))
    else:
        # not support. use padding-zero and pool2d.
        print("Warning: use zero-padding in avgpool")
        outputs = fluid.layers.pad(inputs, padding_pixel, 0)
        return fluid.layers.pool2d(
            outputs, kernel, 'avg', strides, pool_padding=0, ceil_mode=False)


def global_avgpool(inputs):
    return fluid.layers.pool2d(
        inputs,
        1,
        'avg',
        1,
        pool_padding=0,
        global_pooling=True,
        ceil_mode=True)


def fully_connected(inputs, units, act=None, name=None):
    n = inputs.shape[1]
    param_attr = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.NormalInitializer(
            0.0, scale=np.sqrt(2.0 / n)),
        regularizer=fluid.regularizer.L2Decay(flag_weight_decay))

    bias_attr = fluid.param_attr.ParamAttr(
        regularizer=fluid.regularizer.L2Decay(0.))
    
    return fluid.layers.fc(inputs,
                            units,
                            param_attr=param_attr,
                            bias_attr=bias_attr,
                            act=act,
                            name=name)


def bn(inputs, is_test=False, act=None, name=None):
    # batch norm

    return fluid.layers.batch_norm(
        inputs, is_test=is_test, epsilon=0.001, data_layout="NCHW", act=act, name=name+'_bn')


def PixelShuffle(inputs, scale=2):
    size = inputs.shape
    batch_size = size[0]
    c = size[1]
    h = size[2]
    w = size[3]
    
    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    # print("c:",c)
    # print("channel_target:",channel_target)
    # print("inputs:",inputs)
    channel_target=int(channel_target)
    shape_1 = [batch_size, channel_factor // scale, channel_factor // scale, h, w]
    shape_2 = [batch_size, 1, h * scale, w * scale]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = fluid.layers.split(inputs, num_or_sections=channel_target, dim=1)
    
    output = fluid.layers.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=1)

    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = fluid.layers.reshape(inputs, shape_1)
    X = fluid.layers.transpose(X, [0, 1, 3, 2, 4])

    return fluid.layers.reshape(X, shape_2)

# need implement!!
def SubpixelConv_relu(n, scale=2, name=None):
    # pixelshuffle + relu
    output = PixelShuffle(n)
    return fluid.layers.relu(output, name=name)

# need implement!!
def UpSampling2dLayer(n, out_shape=None, method='NEAREST', name=None):
    output = fluid.layers.image_resize(n, out_shape=out_shape, resample=method, name=name)
    return output

def dropout(inputs):
    """ dropout layer """

    return fluid.layers.dropout(inputs, dropout_prob=FLAGS.dropout_rate)
