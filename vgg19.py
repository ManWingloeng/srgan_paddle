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
        VGG_MEAN = [103.939, 116.779, 123.68]
        rgb_scaled = input * 255.0
        # print(input)
        red, green, blue = fluid.layers.split(rgb_scaled, num_or_sections=3, dim=1)
        # print("red: ",red)
        bgr = fluid.layers.concat(input=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],], axis=1)
        # print("rgb_scaled:",rgb_scaled)
        # print('bgr:',bgr)
        y = bgr
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


# ### test for vgg preprocess 
# import paddle.v2 as paddle
# import paddle.fluid as fluid
# from data_reader import *
# THR_reader = paddle.batch(train_hr_reader(), 2)()
# data=next(THR_reader)

# data_thr = map(lambda x: x[0], data)
# data_tlr = map(lambda x: x[1], data)
# data_thr=np.array(data_thr)
# data_thr=np.squeeze(data_thr)

# data_tlr=np.array(data_tlr)
# data_tlr=np.squeeze(data_tlr)
# infer_program = fluid.default_main_program().clone(for_test=True)
# with fluid.program_guard(infer_program):
#     data = fluid.layers.data(name='data',shape=[3, 384, 384])
#     # data_lr = fluid.layers.image_resize(input=data, out_shape=[96, 96], resample='')
#     data_lr = fluid.layers.data(name='data_lr', shape=[3, 96, 96])
#     t_target_image_224 = fluid.layers.resize_bilinear(data, out_shape=[224, 224])  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
#     t_predict_image_224 = fluid.layers.resize_bilinear(data_lr, out_shape=[224, 224])  # resize_generate_image_for_vgg

#     vgg_target_emb = VGG19().net((t_target_image_224 + 1) / 2)
#     vgg_predict_emb = VGG19().net((t_predict_image_224 + 1) / 2)
#     # print("data_lr_tensor:",data_lr.shape)
#     predict = fluid.layers.fc(input=data, size=1, act='relu')
#     softmax = fluid.layers.softmax(predict)
#     print(data.shape)

# place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
# exe = fluid.Executor(place)
# exe.run(fluid.default_startup_program())
# # fluid.io.load_params(exe, "/home/aistudio/work/data/vgg_pd_params")

# p = exe.run(infer_program, fetch_list=[predict, softmax], feed={
#     'data': data_thr,
#     'data_lr': data_tlr
# })


