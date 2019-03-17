from matplotlib import pyplot as plt
import numpy as np


__BASE_DIR_PATH = '/Volumes/SHARE_DRIVE/Machine_Learning/DataSet/cifar-10-batches-py/'
__CIFAR_DATA = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
__LABEL_NAME = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
__TARGET_DATA = 1


# CIFARデータ（binary）取得メソッド
def unpickle(path):
    """
    :param path: cifarファイルのパス
    :return:
    """
    import pickle
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# データ表示メソッド
def showImage(start, end):
    """
    :param start: 表示するデータの最初のインデックス
    :param end: 表示するデータの最後のインデックス
    :return:
    """
    num = end - start
    fig = plt.figure(figsize=(10, 10))
    for i in range(num):
        sub_p = fig.add_subplot(num // 10 + 1, 10, i + 1)
        sub_p.set_xticks([])
        sub_p.set_yticks([])
        sub_p.set_title(__LABEL_NAME[getLabel(i)])
        sub_p.imshow(getImageA(i))
    fig.show()


# 画像取得メソッド
def getImageA(index):
    cifar = unpickle(__BASE_DIR_PATH + __CIFAR_DATA[__TARGET_DATA])
    base_img = cifar[b'data'][index]
    img = cifarImgToImg(base_img)
    return img.astype(np.int32)


# ラベル取得メソッド
def getLabel(index):
    cifar = unpickle(__BASE_DIR_PATH + __CIFAR_DATA[__TARGET_DATA])
    label = cifar[b'labels'][index]
    return label


# cifarの画像データ（R*1024, G*1024, B*1024）を通常の画像データ（RGB * 1024）に変換
def cifarImgToImg(cifarImg):
    tmp_img = np.zeros((3, 1024))
    tmp_img[0] = cifarImg[:1024]
    tmp_img[1] = cifarImg[1024:2048]
    tmp_img[2] = cifarImg[2048:3072]
    img = tmp_img.T.reshape([32, 32, 3])
    return img