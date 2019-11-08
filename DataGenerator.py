import numpy as np
import keras
import cv2
import os

img_rows = 200
img_cols = 200
img_frames = 20


def pre_dealing_data(train_set):
    train_set = train_set.astype('float32')
    train_set -= np.mean(train_set)
    train_set /= np.max(train_set)
    return train_set


def read_img_data(data_list, index, batch_size, main_path):
    categories = data_list[index:(index+batch_size)]
    training_set = np.zeros(shape=(1, img_frames, img_rows, img_cols, 3))
    for index, package in enumerate(categories):
        package_path = os.path.join(main_path, package)
        one_package_vector = np.zeros(shape=(1, img_rows, img_cols, 3))
        for root, dirs, files in os.walk(package_path):
            for innerIndex, file in enumerate(files):
                file_path = os.path.join(package_path, file)
                # 使用opencv读取图片
                img_readed = cv2.imread(file_path)
                # 给图片张量添加帧数维度，并且给不足19帧的图片集用0补足19帧
                img_reshaped = img_readed.reshape(1, img_readed.shape[0], img_readed.shape[1], img_readed.shape[2])
                if innerIndex == 0:
                    one_package_vector = img_reshaped
                elif innerIndex < img_frames:
                    one_package_vector = np.concatenate((one_package_vector, img_reshaped), axis=0)
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
    return training_set


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=20, data_list=None, label_list=None, main_path=None):
        self.batch_size = batch_size
        self.data_list = data_list
        self.label_list = label_list
        self.main_path = main_path

    def __len__(self):
        return len(os.listdir(self.data_list)) // self.batch_size

    def __getitem__(self, idx):
        print(idx)
        x, y = self.__generate_data(idx)
        return x, y

    def __generate_data(self, index):
        training_data = read_img_data(self.data_list, index, self.batch_size, self.main_path)
        training_label = self.label_list[index*self.batch_size:(index+1)*self.batch_size]
        return training_data, training_label
