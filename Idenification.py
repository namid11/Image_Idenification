from DataOperation import *
from img_edit import *



# showImage(0, 100)
#
# # トレーニングデータの数
# __TRAINING_DATA_NUM = 10000
# # トレーニングデータ格納配列（のち、numpy.arrayに変換）
# training_datas = []
# for i in range(__TRAINING_DATA_NUM):
#     training_datas.append(getImageA(i))
# training_datas = np.array(training_datas)



# １段目の畳み込みフィルターとプーリング層を定義
num_filters_1 = 8

x = tf.placeholder(tf.float32, [None, 1728])
x_image = tf.reshape(x, [-1,24,24,3])

w_conv_1 = tf.Variable(tf.truncated_normal([5,5,3,num_filters_1], stddev=0.1))
h_conv_1 = tf.nn.conv2d(x_image, w_conv_1, strides=[1,1,1,1], padding="SAME")                   # 畳み込み処理

b_conv_1 = tf.Variable(tf.constant(0.1, shape=[num_filters_1]))
h_conv_1_cutoff = tf.nn.relu(h_conv_1 + b_conv_1)

h_pool_1 = tf.nn.max_pool(h_conv_1_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")  # プーリング処理


# ２段目の畳み込みフィルターとプーリング層を定義
num_filters_2 = 16

w_conv_2 = tf.Variable(tf.truncated_normal([5,5,num_filters_1, num_filters_2]))
h_conv_2 = tf.nn.conv2d(h_pool_1, w_conv_2, strides=[1,1,1,1], padding="SAME")                  # 畳み込み処理

b_conv_2 = tf.Variable(tf.constant(0.1, shape=[num_filters_2]))
h_conv_2_cutoff = tf.nn.relu(h_conv_2 + b_conv_2)

h_pool_2 = tf.nn.max_pool(h_conv_2_cutoff, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")  # プーリング処理


### 全結合層、ソフトマックス関数を定義
h_pool_2_flat = tf.reshape(h_pool_2, [-1, 6*6*num_filters_2])

num_units_1 = 6*6*num_filters_2
num_units_2 = 1024

# 全結合層（隠れ層）
w2 = tf.Variable(tf.truncated_normal([num_units_1, num_units_2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units_2]))
hidden_2 = tf.nn.relu(tf.matmul(h_pool_2_flat, w2) + b2)
# ソフトマックス関数
w0 = tf.Variable(tf.zeros([num_units_2, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden_2, w0) + b0)


# 誤差関数 loss, トレーニングアルゴリズム, 正解率を定義
t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# セーバーインスタンス作成
saver = tf.train.Saver()

# トレーニング実行
if __name__ == '__main__':
    with tf.Session() as sess:
        
        # チェックポイントの確認
        ckpt_state = tf.train.get_checkpoint_state("./sess_data")
        if ckpt_state:
            # チェックポイントあれば、variableを取得
            restore_model = ckpt_state.model_checkpoint_path
            saver.restore(sess, restore_model)
        else:
            # チェックポイントがなければ、トレーニング
            
            sess.run(tf.global_variables_initializer())
            batch_size = 30

            t_num = 1000
            for i in range(t_num):
                # トレーニングデータ取得
                set_target_data(1)
                batch_imgs = get_frames_data(sess, imgs_processing(getImagesArray(i*batch_size % t_num, i*batch_size % t_num + batch_size)))
                batch_imgs = batch_imgs.reshape([-1, 1728])
                batch_labels = getLabelsArray(i*batch_size % t_num, i*batch_size % t_num + batch_size, 10)
                sess.run(train_step, feed_dict={x:batch_imgs, t:batch_labels})

                # 正解率確認
                if (i+1) % 100 == 0:
                    set_target_data(2)
                    acc_val = sess.run(accuracy, feed_dict={x:get_frames_data(sess, imgs_crop(getImagesArray(0, 100))).reshape([-1, 1728]),
                                                            t:getLabelsArray(0, 100, 10)})
                    print('Step: %d, Accuracy: %f' % (i+1, acc_val))
                    saver.save(sess, './sess_data/sess.ckpt', global_step=i+1)
                else:
                    print('Step: %d' % (i+1))
