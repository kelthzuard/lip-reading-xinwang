# lip-reading-xinwang
# 2019/10/15 汪腾睿
更新图片处理脚本img-cutting，把原来图像截取为中间部分的灰度图像。
未对其他大量数据集样本做测试，可能还存在很大的问题。
提交代码注意不要把图片带上来，把图片文件写进gitignore里面
# 2019/10/22 汪腾睿
更新数据处理函数read_and_initial_data，具体见代码以及注释
# 2019/10/23 汪腾睿
更新label处理函数，把label按图片顺序排列成一维数组，并且映射成整数后进行one-hot编码
# 2019/11/3  余方剑
新增训练脚本，初步设定为3层卷积，2层LSTM，2层全连接，参数未设置
# 2019/11/4 汪腾睿
加入ConvLstm2D层，并成功训练，还存在着较大的bug，后续再改，还需改造使用generator训练
# 2019/11/5 汪腾睿
图片读取重写为generator形式，使用fit_generator进行训练。