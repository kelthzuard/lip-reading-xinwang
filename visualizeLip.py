import numpy as np
import cv2
import dlib
import math
import sys
import pickle
import argparse
import os
from skimage import io

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video file")
ap.add_argument("-o", "--output", required=True,  #使用相对路径  ./results
                help="path to output video file")
args = vars(ap.parse_args())

predictor_path = './data/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mouth_destination_path = os.path.dirname(args["output"]) + '/' + 'mouth'
if not os.path.exists(mouth_destination_path):
    os.makedirs(mouth_destination_path)
imgReaded = cv2.imread(args["input"])
img_gray = cv2.cvtColor(imgReaded, cv2.COLOR_RGB2GRAY)
activation = 0

# Detection of the frame
# 实例化检测器
detections = detector(img_gray, 0)
landmarks = np.matrix([[p.x, p.y] for p in predictor(imgReaded,detections[0]).parts()])
#print(landmarks, type(landmarks))
for idx, point in enumerate(landmarks):
    # 68点的坐标
    pos = (point[0, 0], point[0, 1])
    print(idx+1, pos)

# 20 mark for mouth
# 初始化一个两行十列的0元素数组用于嘴部部分标记
marks = np.zeros((2, 20))
