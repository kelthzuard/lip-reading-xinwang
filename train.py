import cv2
import os
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input file categories")
args = vars(ap.parse_args())
main_path = args["input"]
categories = os.listdir(main_path)

traning_set = np.zeros(shape=(1, 15, 200, 200, 3))
for package in categories:
    package_path = os.path.join(main_path, package)
    one_package_vector = np.zeros(shape=(1, 200, 200, 3))
    for root, dirs, files in os.walk(package_path):
        for file in files:
            file_path = os.path.join(package_path, file)

            img_readed = cv2.imread(file_path)
            img_reshaped = img_readed.reshape(1, img_readed.shape[0], img_readed.shape[1], img_readed.shape[2])
            one_package_vector = np.concatenate((one_package_vector, img_reshaped), axis=0)
    one_package_vector_reshaped = one_package_vector.reshape(1, one_package_vector.shape[0], one_package_vector.shape[1], one_package_vector.shape[2], one_package_vector.shape[3])
    traning_set = np.concatenate((traning_set, one_package_vector_reshaped), axis=0)