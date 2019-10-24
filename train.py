import cv2
import os
import argparse
import numpy as np
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input file categories")
ap.add_argument("-l", "--label", required=True,
                help="path to input label")
args = vars(ap.parse_args())


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


# 将label中的中文单词进行序列化并进行one-hot编码
def encode_label(label):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(label)
    sequences = tokenizer.texts_to_sequences(label)
    one_hot_results = tokenizer.texts_to_matrix(label, mode='binary')
    return one_hot_results, tokenizer


# 解码
def decode_label(label, tokenizer):
    # 这个地方怎么去反向解码好像灭有找到api，那就只有从sequences去操作吗还是怎么的
    # 太晚了，以后再弄，应该不急着用
    return 0


# 输入数据集文件夹的一级目录，导出一个5D张量数据
# (samples, frame, rows, cols, channels)
# 此张量作为导入3d卷积神经网络的参数格式
def read_and_initial_data(main_path, label_list):
    # 初始化一维标签数组
    label = []
    categories = os.listdir(main_path)
    training_set = np.zeros(shape=(1, img_frames, img_rows, img_cols, 3))
    for index, package in enumerate(categories):
        package_path = os.path.join(main_path, package)
        one_package_vector = np.zeros(shape=(1, img_rows, img_cols, 3))
        for root, dirs, files in os.walk(package_path):
            for innerIndex, file in enumerate(files):
                file_path = os.path.join(package_path, file)
                # 使用opencv读取图片
                img_readed = cv2.imread(file_path)
                # 给图片张量添加帧数维度，并且给不足15帧的图片集用0补足15帧
                img_reshaped = img_readed.reshape(1, img_readed.shape[0], img_readed.shape[1], img_readed.shape[2])
                if innerIndex == 0:
                    one_package_vector = img_reshaped
                else:
                    one_package_vector = np.concatenate((one_package_vector, img_reshaped), axis=0)
        # 把sample对应的label添加进label数组中
        label.append(label_list[package])
        while one_package_vector.shape[0] < img_frames:
            padding_vector = np.zeros(shape=(1, img_rows, img_cols, 3))
            one_package_vector = np.concatenate((one_package_vector, padding_vector), axis=0)
        one_package_vector_reshaped = one_package_vector.reshape(1, one_package_vector.shape[0],
                                                                 one_package_vector.shape[1],
                                                                 one_package_vector.shape[2],
                                                                 one_package_vector.shape[3])
        if index == 0:
            training_set = one_package_vector_reshaped
        else:
            training_set = np.concatenate((training_set, one_package_vector_reshaped), axis=0)
    training_set = pre_dealing_data(training_set)
    return training_set, label


# 预处理数据
def pre_dealing_data(train_set):
    train_set = train_set.astype('float32')
    train_set -= np.mean(train_set)
    train_set /= np.max(train_set)
    return train_set


label_list = read_and_initial_label((args["label"]))
training_data, label = read_and_initial_data(args["input"], label_list)
label = encode_label(label)

# *****************
# 定义3D卷积网络模型参数
# *****************
img_rows = 200
img_cols = 200
img_frames = 15
patch_size = 15  # 每批次训练数量
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

    model.add(Convolution3D(
        filters_3D[0],
        kernel_dim1=conv_3D[0],  # depth
        kernel_dim2=conv_3D[0],  # rows
        kernel_dim3=conv_3D[0],  # cols
        input_shape=(3, img_rows, img_cols, patch_size),
        activation='relu'
    ))

    model.add(MaxPooling3D(pool_size=(pool_3D[0], pool_3D[0], pool_3D[0])))

    model.add(Convolution3D(
        filters_3D[1],
        kernel_dim1=conv_3D[1],  # depth
        kernel_dim2=conv_3D[1],  # rows
        kernel_dim3=conv_3D[1],  # cols
        activation='relu'
    ))

    model.add(MaxPooling3D(pool_size=(pool_3D[1], pool_3D[1], pool_3D[1])))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(128, init='normal', activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(6, init='normal'))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])

