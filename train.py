import cv2
import os
import argparse
import numpy as np
from keras.preprocessing.text import Tokenizer

from DataGenerator import DataGenerator

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import ConvLSTM2D, LSTM

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
                help="path to input file categories")
ap.add_argument("-l", "--label", required=False,
                help="path to input label")
args = vars(ap.parse_args())
input_path = args["input"]
label_path = args["label"]

# **********
# 定义参数
# **********
img_rows = 200
img_cols = 200
img_frames = 21
train_samples = 8000
label_out_put = None
batch_size = 5  # 每批次训练数量


# 从lip_train.txt读取label对应值进一个dict里
def read_and_initial_label(file_path):
    label = {}
    f = open(file_path, "r", encoding="utf-8")
    line = f.readline()
    line = line[:-1]
    while line:
        label_array = line.split()
        label[label_array[0]] = label_array[1]
        line = f.readline()
        line = line[:-1]
    f.close()
    return label


def reform_label_order(main_path, label_list):
    label = []
    categories = os.listdir(main_path)
    for index, package in enumerate(categories):
        label.append(label_list[package])
    return label


def encode_label(label):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(label)
    sequences = tokenizer.texts_to_sequences(label)
    one_hot_results = tokenizer.texts_to_matrix(label, mode='binary')
    return one_hot_results


label_list = read_and_initial_label(label_path)
label = reform_label_order(input_path, label_list)
label = encode_label(label)
label_out_put = len(label[0])

# 划分训练集和测试集
train_categories = os.listdir(input_path)[:train_samples]
train_label = label[:train_samples]
training_generation = DataGenerator(batch_size=batch_size,
                                    data_list=train_categories,
                                    label_list=train_label,
                                    main_path=input_path)
validation_categories = os.listdir(input_path)[(train_samples+1):]
validation_label = label[(train_samples+1):]
validation_generation = DataGenerator(batch_size=batch_size,
                                      data_list=validation_categories,
                                      label_list=validation_label,
                                      main_path=input_path)

# *****************
# 定义3D卷积网络模型参数
# *****************
filters_3D = [32, 50]  # 卷积核数量
conv_3D = [5, 5]  # 卷积核尺寸
pool_3D = [3, 3]  # 池化层尺寸

# 加载模型
model_exists = os.path.exists('model.h5')
if (model_exists):
    model = load_model('model.h5')
    print("**************************************************")
    print("model.h5 model loaded")

else:
    model = Sequential()

    model.add(Conv3D(
        filters_3D[0],
        (conv_3D[0], conv_3D[0], conv_3D[0]),
        input_shape=(img_frames, img_rows, img_cols, 3),
        activation='relu'
    ))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), data_format="channels_last"))

    model.add(Conv3D(
        filters_3D[1],
        (conv_3D[0], conv_3D[0], conv_3D[0]),
        activation='relu'
    ))

    model.add(MaxPooling3D(pool_size=(pool_3D[1], pool_3D[1], pool_3D[1]), data_format="channels_last"))

    model.add(Dropout(0.5))

    model.add(ConvLSTM2D(
        filters=40, kernel_size=(3, 3), padding="same"
    ))

    model.add(Flatten())

    model.add(Dense(128, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(label_out_put, kernel_initializer='normal'))  # 这个地方的输出维度需要和最终的分类维度相同，需要根据数据集的不同指定

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    model.fit_generator(
                        generator=training_generation,
                        validation_data=validation_generation,
                        samples_per_epoch=train_samples,
                        epochs=1
                        )
