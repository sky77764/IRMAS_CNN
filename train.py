from __future__ import division, print_function, absolute_import

import tensorflow as tf
from transform import transformMEL
import sys, os 
import util
import numpy as np
import scipy 
import scipy.io.wavfile
import matplotlib 
import matplotlib.pyplot as plt
import dataset
from dataset import MyDataset
from datetime import datetime
path_to_irmas = '/home/js/dataset/IRMAS/'
feature_dir_train = os.path.join(path_to_irmas,'features','Training')
if not os.path.exists(feature_dir_train):
    print(feature_dir_train + ' does not exist!')
    exit()

d=os.path.join(path_to_irmas,'Training')
instruments = sorted(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))

ckpt_path = './ckpt/train_model_no_dropout_softmax.ckpt'
ckpt_load_idx = 0

# Training Parameters
learning_rate = 0.001
num_steps = 200000
batch_size = 128

db=MyDataset(feature_dir=feature_dir_train, batch_size=batch_size, time_context=128, step=50, 
             suffix_in='_mel_',suffix_out='_label_',floatX=np.float32,train_percent=0.85)

# Network Parameters
feature_dim = 5504 
num_classes = 11 
dropout = 1.0 # 0.75 
dropout2 = 1.0 # 0.5

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def zero_pad(x, k=2):
    return tf.pad(x, [[0, 0], [k, k], [k, k], [0, 0]], "CONSTANT")

def cnn(x, weights, biases, keep_prob, keep_prob2):
    # 128 x 43 x 1
    x = tf.reshape(x, shape=[-1, 43, 128, 1])
    x = zero_pad(x, k=2)
    conv1 = conv2d(x, weights['w1_1'], biases['bw1_1'])
    conv1 = zero_pad(conv1, k=2)
    conv1 = conv2d(conv1, weights['w1_2'], biases['bw1_2'])
    conv1 = maxpool2d(conv1, k=3)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # print(conv1.get_shape())

    # 44 x 15 x 32
    conv1 = zero_pad(conv1, k=2)
    conv2 = conv2d(conv1, weights['w2_1'], biases['bw2_1'])
    conv2 = zero_pad(conv2, k=2)
    conv2 = conv2d(conv2, weights['w2_2'], biases['bw2_2'])
    conv2 = maxpool2d(conv2, k=3)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # print(conv2.get_shape())

    # 16 x 6 x 64
    conv2 = zero_pad(conv2, k=2)
    conv3 = conv2d(conv2, weights['w3_1'], biases['bw3_1'])
    conv3 = zero_pad(conv3, k=2)
    conv3 = conv2d(conv3, weights['w3_2'], biases['bw3_2'])
    conv3 = maxpool2d(conv3, k=3)
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # print(conv3.get_shape())

    # 6 x 3 x 128
    conv3 = zero_pad(conv3, k=2)
    conv4 = conv2d(conv3, weights['w4_1'], biases['bw4_1'])
    conv4 = zero_pad(conv4, k=2)
    conv4 = conv2d(conv4, weights['w4_2'], biases['bw4_2'])
    conv4 = tf.layers.max_pooling2d(conv4, 7, 10)
    # print(conv4.get_shape())

    # 1 x 1 x 256
    fc1 = tf.contrib.layers.flatten(conv4)
    fc1 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, weights['fc1']), biases['bfc1']))
    fc1 = tf.nn.dropout(fc1, keep_prob2)
    # print(fc1.get_shape())

    # 1024
    out = tf.add(tf.matmul(fc1, weights['fc2']), biases['bfc2'])
    return out

weights = {
    'w1_1' : tf.Variable(tf.random_normal([3, 3, 1, 32]), name='w1_1'),
    'w1_2' : tf.Variable(tf.random_normal([3, 3, 32, 32]), name='w1_2'),
    'w2_1' : tf.Variable(tf.random_normal([3, 3, 32, 64]), name='w2_1'),
    'w2_2' : tf.Variable(tf.random_normal([3, 3, 64, 64]), name='w2_2'),
    'w3_1' : tf.Variable(tf.random_normal([3, 3, 64, 128]), name='w3_1'),
    'w3_2' : tf.Variable(tf.random_normal([3, 3, 128, 128]), name='w3_2'),
    'w4_1' : tf.Variable(tf.random_normal([3, 3, 128, 256]), name='w4_1'),
    'w4_2' : tf.Variable(tf.random_normal([3, 3, 256, 256]), name='w4_2'),
    'fc1' : tf.Variable(tf.random_normal([256, 1024]), name='fc1'),
    'fc2' : tf.Variable(tf.random_normal([1024, num_classes]), name='fc2')
}

biases = {
    'bw1_1' : tf.Variable(tf.random_normal([32]), name='bw1_1'),
    'bw1_2' : tf.Variable(tf.random_normal([32]), name='bw1_2'),
    'bw2_1' : tf.Variable(tf.random_normal([64]), name='bw2_1'),
    'bw2_2' : tf.Variable(tf.random_normal([64]), name='bw2_2'),
    'bw3_1' : tf.Variable(tf.random_normal([128]), name='bw3_1'),
    'bw3_2' : tf.Variable(tf.random_normal([128]), name='bw3_2'),
    'bw4_1' : tf.Variable(tf.random_normal([256]), name='bw4_1'),
    'bw4_2' : tf.Variable(tf.random_normal([256]), name='bw4_2'),
    'bfc1': tf.Variable(tf.random_normal([1024]), name='bfc1'),
    'bfc2': tf.Variable(tf.random_normal([num_classes]), name='bfc2')
}

x = tf.placeholder(tf.float32, [None, feature_dim])
y_ = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
y_conv = cnn(x, weights, biases, keep_prob, keep_prob2)


loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_conv, labels=tf.cast(y_, dtype=tf.float32)))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

train_images = db.features.reshape(-1, feature_dim)
train_labels = np.zeros((db.labels.shape[0], num_classes))
train_labels[np.arange(db.labels.shape[0]), db.labels[:,0].astype(int)] = 1


valid_images = db.features_valid.reshape(-1, feature_dim)
valid_labels = np.zeros((db.labels_valid.shape[0], num_classes))
valid_labels[np.arange(db.labels_valid.shape[0]), db.labels_valid[:,0].astype(int)] = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if ckpt_load_idx != 0:
        saver.restore(sess, ckpt_path + '-'+ str(ckpt_load_idx))
        print("Model restored.")

    for i in range(num_steps):
        rand_index = np.random.choice(train_images.shape[0], size=batch_size)
        x_batch, y_batch = train_images[rand_index], train_labels[rand_index]  
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: x_batch, y_: y_batch, keep_prob: 1.0, keep_prob2: 1.0})
            print('[%s] step %d, training accuracy %g' % (datetime.now().strftime('%m-%d %H:%M:%S'), i+ckpt_load_idx, train_accuracy))

        if i != 0 and i % 1000 == 0:
            rand_index = np.random.choice(valid_images.shape[0], size=1000)
            x_valid_batch, y_valid_batch = valid_images[rand_index], valid_labels[rand_index]
            valid_accuracy = accuracy.eval(feed_dict={
                x: x_valid_batch, y_: y_valid_batch, keep_prob: 1.0, keep_prob2: 1.0})
            print('[%s] validation accuracy %g' % (datetime.now().strftime('%m-%d %H:%M:%S'), valid_accuracy))
            save_path = saver.save(sess, ckpt_path, global_step=i+ckpt_load_idx)
            print("Model saved in file: %s" % save_path)

        optimizer.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: dropout, keep_prob2: dropout2})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: valid_images, y_: valid_labels, keep_prob: 1.0, keep_prob2: 1.0}))
