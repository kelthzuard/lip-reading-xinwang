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

filters_3D_1 = 64
kernel_size1 = (1,3,3)
stride1 = (1,2,2)

max_pool_size1 = (1,3,3)
max_pool_stride1 = (1,2,2)
max_pool_pad1 = (0,1,1)

#输入图片的情况：200*200*19
img_rows = 200
img_cols = 200
img_frames = 20

#如果模型存在，加载已有模型
if(os.path.exists('model.h5')):
    print('model is loading')
    model = load_model('model.h5')

#否则使用代码里写的模型
else:
    model = Sequential()
    #第一层卷积层
    model.add(Conv3D(filters_3D_1,kernel_size1,stride1,input_shape=(img_frames, img_rows, img_cols, 3),activation="relu"))
    #第一层最大池化层
    model.add(MaxPooling3D(max_pool_size1, strides=max_pool_stride1, padding=max_pool_pad1))
    #第二层卷积层
    model.add(Conv3D())
    #第二层最大池化层
    model.add(MaxPooling3D())
    #第三层卷积层
    model.add(Conv3D(

    ))
    #第三层最大池化层
    model.add(MaxPooling3D())
    #第一层dropout层防止过拟合
    model.add(Dropout())
    #第一层LSTM层
    model.add(LSTM())
    #第二层LSTM层
    model.add(LSTM())
    #第二层dropout层防止过拟合
    model.add(Dropout())
    #第一层Flatten层
    model.add(Flatten())
    #第一层全连接层
    model.add(Dense())
    #第二层全连接层
    model.add(Dense())
    #softmax方法进行学习
    model.add(Activation('softmax'))

