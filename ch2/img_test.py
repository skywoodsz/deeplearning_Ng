#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'skywoodsz'

import numpy as np
import matplotlib.pyplot as plt # matlab 绘图
import h5py # 数据集库
from lr_utils import load_dataset #导入H5数据集
import scipy # 数学库
from PIL import Image #py图像处理库
from scipy import ndimage

my_image = "azu.png"
fname = "image/" + my_image
img = np.array(ndimage.imread(fname,flatten=False))
plt.imshow(img)
plt.show()