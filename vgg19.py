import paddle.v2 as paddle
import paddle.fluid as fluid
import numpy as np


class VGG19():

    def __init__(self, include_top=False, infer=True):
        self.include_top = include_top
        self.infer = infer

    def _conv2d(self, input, num_filters, name):
        return fluid.layers.conv2d(
            input,
            num_filters=num_filters,
            filter_size=3,
            padding=1,
            param_attr=name + "_weights",
            bias_attr=name + "_biases",
            act='relu')

    def _maxpool(self, input):
        return fluid.layers.pool2d(
            input,
            pool_size=2,
            pool_stride=2,
            ceil_mode=True)

    def _fc(self, input, size, act, name=""):
        return fluid.layers.fc(
            input,
            size=size,
            act=act,
            param_attr=name + "_weights",
            bias_attr=name + "_biases")



    def net(self, input):
        # input = self.load_data(im, im.shape[1:])
        y = input
        y = self._conv2d(y, num_filters=64, name="conv1_1")
        y = self._conv2d(y, num_filters=64, name="conv1_2")
        y = self._maxpool(y)

        y = self._conv2d(y, num_filters=128, name="conv2_1")
        y = self._conv2d(y, num_filters=128, name="conv2_2")
        y = self._maxpool(y)

        y = self._conv2d(y, num_filters=256, name="conv3_1")
        y = self._conv2d(y, num_filters=256, name="conv3_2")
        y = self._conv2d(y, num_filters=256, name="conv3_3")
        y = self._conv2d(y, num_filters=256, name="conv3_4")
        y = self._maxpool(y)

        y = self._conv2d(y, num_filters=512, name="conv4_1")
        y = self._conv2d(y, num_filters=512, name="conv4_2")
        y = self._conv2d(y, num_filters=512, name="conv4_3")
        y = self._conv2d(y, num_filters=512, name="conv4_4")
        y = self._maxpool(y)

        y = self._conv2d(y, num_filters=512, name="conv5_1")
        y = self._conv2d(y, num_filters=512, name="conv5_2")
        y = self._conv2d(y, num_filters=512, name="conv5_3")
        y = self._conv2d(y, num_filters=512, name="conv5_4")
        y = self._maxpool(y)

        if not self.include_top:
            return y

        y = self._fc(y, size=4096, name="fc6", act='relu')
        if not self.infer:
            y = fluid.layers.dropout(y, dropout_prob=0.5)

        y = self._fc(y, size=4096, name="fc7", act='relu')
        if not self.infer:
            y = fluid.layers.dropout(y, dropout_prob=0.5)

        y = self._fc(y, size=1000, name="fc8", act='softmax')

        return y