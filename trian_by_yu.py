import cv2
import os
import argparse
import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import LSTM

print("ok,let's go!")

filters_3D_1 = 0
kernel_1 = 0
kernel_2 = 0

#如果模型存在，加载已有模型
if(os.path.exists('model.h5')):
    print('model is loading')
    model = load_model('model.h5')

#否则使用代码里写的模型
else:
    model = Sequential()
    #第一层卷积层
    model.add(Conv3D(
        filters_3D_1,
        (kernel_1,kernel_2)
    ))

    model.add(MaxPooling3D())

    model.add(Conv3D(

    ))

    model.add(MaxPooling3D())

    model.add(Conv3D(

    ))

    model.add(MaxPooling3D())

    model.add(Dropout())

    model.add(LSTM())

    model.add(LSTM())

    model.add(Dropout())

    model.add(Flatten())

    model.add(Dense())

    model.add(Activation('softmax'))

