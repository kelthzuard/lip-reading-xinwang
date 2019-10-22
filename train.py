import cv2
import os
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input file categories")
args = vars(ap.parse_args())

#输入数据集文件夹的一级目录，导出一个5D张量数据
#(samples, frame, rows, cols, channels)
#此张量作为导入3d卷积神经网络的参数格式
def read_and_initial_data (main_path):
    categories = os.listdir(main_path)
    traning_set = np.zeros(shape=(1, 15, 200, 200, 3))
    for index, package in enumerate(categories):
        package_path = os.path.join(main_path, package)
        one_package_vector = np.zeros(shape=(1, 200, 200, 3))
        for root, dirs, files in os.walk(package_path):
            for innerIndex, file in enumerate(files):
                file_path = os.path.join(package_path, file)
                #使用opencv读取图片
                img_readed = cv2.imread(file_path)
                #给图片张量添加帧数维度，并且给不足15帧的图片集用0补足15帧
                img_reshaped = img_readed.reshape(1, img_readed.shape[0], img_readed.shape[1], img_readed.shape[2])
                if innerIndex == 0:
                    one_package_vector = img_reshaped
                else:
                    one_package_vector = np.concatenate((one_package_vector, img_reshaped), axis=0)
        while one_package_vector.shape[0] < 15:
            padding_vector = np.zeros(shape=(1, 200, 200, 3))
            one_package_vector = np.concatenate((one_package_vector, padding_vector), axis=0)
        one_package_vector_reshaped = one_package_vector.reshape(1, one_package_vector.shape[0],
                                                                 one_package_vector.shape[1],
                                                                 one_package_vector.shape[2],
                                                                 one_package_vector.shape[3])
        if index == 0:
            traning_set = one_package_vector_reshaped
        else:
            traning_set = np.concatenate((traning_set, one_package_vector_reshaped), axis=0)
    return traning_set


traning_dataset = read_and_initial_data(args["input"])