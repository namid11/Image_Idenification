import numpy as np
import tensorflow as tf
from DataOperation import *


showImage(0, 100)

# トレーニングデータの数
__TRAINING_DATA_NUM = 1000
# トレーニングデータ格納配列（のち、numpy.arrayに変換）
training_datas = []
for i in range(__TRAINING_DATA_NUM):
    training_datas.append(getImageA(i))
training_datas = np.array(training_datas)

