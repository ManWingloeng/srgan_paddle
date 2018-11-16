from __future__ import absolute_import

import paddle.fluid as fluid
import paddle.fluid.io as io
import numpy as np
import sys
from PIL import Image
from keras.applications.vgg16 import decode_predictions

from vgg19 import VGG19

## Read image and preprocess
im = Image.open(sys.argv[1])

im = im.resize((224, 224), Image.ANTIALIAS)
im = np.array(im).astype(np.float32)
im = im.transpose((2, 0, 1))  # CHW
im = im[(2, 1, 0), :, :]  # BGR

mean = np.array([104., 117., 124.], dtype=np.float32)
mean = mean.reshape([3, 1, 1])
im = im - mean
im = np.expand_dims(im, 0)

## Define the network
infer_program = fluid.default_main_program().clone(for_test=True)
with fluid.program_guard(infer_program):
    inputs = fluid.layers.data('img', shape=[3, 224, 224])
    predict = VGG19(include_top=True, infer=True).net(inputs)

## Create executor
# place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
## Load params
io.load_params(exe, "./vgg19_pd_params")

# ## Create inference program
# infer_program = fluid.default_main_program().clone(for_test=True)
# infer_program.prune(predict)

## Run it
p = exe.run(infer_program, fetch_list=[predict], feed={
    'img': im
})

p = decode_predictions(p[0])

for (_, cls, prob) in p[0]:
    print("{}: {}".format(cls, prob))
