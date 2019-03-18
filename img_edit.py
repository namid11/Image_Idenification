import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


# 画像の一部を摘出
def imgs_crop(imgs):
    crop_frames = []
    for i, img in enumerate(imgs):
        # 画像の一部を摘出
        crop_frame = tf.random_crop(img, [700, 500, 3])
        crop_frames.append(crop_frame)
    return np.array(crop_frames)


# 画像をランダムで反転
def imgs_flip_lr(imgs):
    flip_frames = []
    for i, img in enumerate(imgs):
        flip_frame = tf.image.random_flip_left_right(img)
        flip_frames.append(flip_frame)
    return np.array(flip_frames)


# 画像の明度をランダム加工
def imgs_ran_brightness(imgs):
    ran_bright_frames = []
    for i, img in enumerate(imgs):
        ran_bright_frame = tf.image.random_brightness(img, 0.6)
        ran_bright_frames.append(ran_bright_frame)
    return np.array(ran_bright_frames)


# 画像のコントラストをランダム加工
def imgs_ran_contrast(imgs):
    ran_contrast_frames = []
    for i, img in enumerate(imgs):
        ran_contrast_frame = tf.image.random_contrast(img, 0.5, 1.5)
        ran_contrast_frames.append(ran_contrast_frame)
    return np.array(ran_contrast_frames)


# 画像標準化モデルの作成
def imgs_std(imgs):
    std_frames = []
    for i, img in enumerate(imgs):
        # 画像を標準化
        std_frame = tf.image.per_image_standardization(img)
        std_frames.append(std_frame)
    return np.array(std_frames)


# 画像加工統合処理（ランダム切り出し->ランダム左右反転->ランダム明度変更->ランダムコントラスト変更->標準化）
def imgs_processing(imgs):
    crop = imgs_crop(imgs)
    flip_lr = imgs_flip_lr(crop)
    ran_brightness = imgs_ran_brightness(flip_lr)
    ran_contrast = imgs_ran_contrast(ran_brightness)
    standardization = imgs_std(ran_contrast)
    return standardization


# 画像保存メソッド
def imgs_save(imgs, path='test'):
    fig = plt.figure(figsize=(5,5))
    for i, img in enumerate(imgs):
        sub_p = fig.add_subplot(1, len(imgs), i+1)
        sub_p.set_xticks([])
        sub_p.set_yticks([])
        sub_p.imshow(img)
    fig.savefig('test_E.jpeg')


def imgs_show(imgs, path='test'):
    fig = plt.figure(figsize=(5,5))
    for i, img in enumerate(imgs):
        sub_p = fig.add_subplot(1, len(imgs), i+1)
        sub_p.set_xticks([])
        sub_p.set_yticks([])
        sub_p.imshow(img)
    fig.show()



if __name__ == '__main__':
    img = cv2.imread('./images/test_1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with tf.Session() as sess:
        for frame in imgs_crop([img]):
            edit_img = sess.run(frame)
            imgs_show([edit_img])