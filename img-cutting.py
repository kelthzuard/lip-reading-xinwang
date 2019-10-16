import cv2
import argparse
import os

#使用命令行把数据集文件带入代码
#示例：python img-cutting.py -i .\data
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input file categories")
args = vars(ap.parse_args())

# 循环目录找到所有文件
for root,dirs,files in os.walk(args["input"]):
    for file in files:
        file_path = os.path.join(root,file)

        imgReaded = cv2.imread(file_path)

        height = len(imgReaded)
        width = len(imgReaded[0])

        cutHeight = int(height*0.2)
        cutWidth = int(height*0.2)

        #去掉图像两边上下20%的部分，仅仅保留中间部分
        cropped = imgReaded[cutHeight:(height-cutHeight), cutWidth:(width-cutWidth)]
        dim = (200, 200)
        resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(file_path, resized)
