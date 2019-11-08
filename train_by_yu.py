import cv2
import os
import argparse
import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D,MaxPooling2D
from keras.layers import LSTM,BatchNormalization,ReLU,AveragePooling3D,ConvLSTM2D

import train
from DataGenerator import DataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
                help="path to input file categories")
ap.add_argument("-l", "--label", required=False,
                help="path to input label")
args = vars(ap.parse_args())
input_path = args["input"]
label_path = args["label"]
print("ok,let's go!")

filters_3D_1 = 64
kernel_size1 = (1,3,3)
stride1 = (1,2,2)

max_pool_size1 = (1,3,3)
max_pool_stride1 = (1,2,2)
max_pool_pad1 = (0,1,1)

bn_size = 4

drop_rate = 0.2

num_classes = 1000

growth_rate = 32

block_config = (4,8,12,8)
#输入图片的情况：200*200*19
img_rows = 200
img_cols = 200
img_frames = 20
batch_size = 1
train_samples = 8000

label_list = train.read_and_initial_label(file_path = args["label"])
label = train.reform_label_order(args["input"],label_list)
label = train.encode_label(label)
label_out_put = len(label[0])

train_categories = os.listdir(input_path)[:train_samples]
train_label = label[:train_samples]
training_generation = DataGenerator(batch_size=batch_size, data_list=args["input"], label_list=label,main_path=input_path)
validation_categories = os.listdir(input_path)[(train_samples+1):]
validation_label = label[(train_samples+1):]
validation_generation = DataGenerator(batch_size=batch_size, data_list=args["input"], label_list=label,main_path=input_path)

def DenseBlock(model,num_layers,num_input_features,bn_size,growth_rate,drop_rate):
    for i in range(num_layers):
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv3D(bn_size*growth_rate,kernel_size=1,strides=1,use_bias=False))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Conv3D(growth_rate,kernel_size=3,strides=1,padding='same',use_bias=False))
    return model

def Transition(model,num_input_features,num_output_features):
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv3D(num_output_features,kernel_size=1,strides=1,use_bias=False))
    model.add(AveragePooling3D(pool_size=(1,2,2),strides=(1,2,2)))
    return model

#如果模型存在，加载已有模型
if(os.path.exists('model.h5')):
    print('model is loading')
    model = load_model('model.h5')

#否则使用代码里写的模型
else:
    model = Sequential()
    #第一层卷积层
    model.add(Conv3D(filters_3D_1,kernel_size1,strides=stride1,input_shape=(img_frames, img_rows, img_cols, 3),
                     activation="relu"))
    #批量标准化层
    model.add(BatchNormalization(axis=2))
    model.add(ReLU())
    #第一层最大池化层
    model.add(MaxPooling3D(pool_size=max_pool_size1, strides=max_pool_stride1, padding="same"))
    num_features = filters_3D_1
    for i,num_layers in enumerate(block_config):
        model = DenseBlock(model=model,num_layers = num_layers,num_input_features=num_features,
                           bn_size=bn_size,growth_rate=growth_rate,drop_rate=drop_rate)
        num_features = num_features + num_features * growth_rate
        if i != len(block_config) - 1:
            model = Transition(model = model,num_input_features=num_features,num_output_features=num_features//2)
            num_features = num_features//2

    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=5,kernel_size=1))
    # model.add(ConvLSTM2D(filters=4, kernel_size=1))
    # model.add(LSTM(536*3*3))
    # model.add(LSTM(512))


    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(label_out_put,kernel_initializer="normal"))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.fit_generator(training_generation,
                    samples_per_epoch=train_samples,
                    epochs=1
                    )
