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
import moviepy.editor as mp

VIDEO_MODE = True
MAKE_CLIP = True

filename = "saxophone"
if VIDEO_MODE:
    audio_file_name = "./sound/"+filename+".wav"
    if MAKE_CLIP:
        clip = mp.VideoFileClip("./video/"+filename+".mp4").subclip(75,90)
        clip.audio.write_audiofile(audio_file_name)
        clip.write_videofile("./video_clip/"+filename+".mp4")
else:
    audio_file_name = './'+filename+'.wav'

path_to_irmas = '/home/js/dataset/IRMAS/'

d=os.path.join(path_to_irmas,'Training')
instruments = sorted(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))

ckpt_path = './ckpt/train_mode_batchnorm3_softmax.ckpt'
ckpt_load_idx = 105000

batch_size = 128

# Network Parameters
feature_dim = 5504 
num_classes = 9 

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def cnn(x, weights, biases, keep_prob, keep_prob2):
    # 128 x 43 x 1
    x = tf.reshape(x, shape=[-1, 43, 128, 1])
    conv1 = conv2d(x, weights['w1_1'], biases['bw1_1'])
    conv1 = maxpool2d(conv1)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    conv1_bn = tf.layers.batch_normalization(conv1, name='conv1_bn')
    # print(conv1.get_shape())

    # 64 x 22 x 128
    conv2 = conv2d(conv1_bn, weights['w2_1'], biases['bw2_1'])
    conv2 = maxpool2d(conv2)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    conv2_bn = tf.layers.batch_normalization(conv2, name='conv2_bn')
    # print(conv2.get_shape())

    # 32 x 11 x 256
    conv3 = conv2d(conv2_bn, weights['w3_1'], biases['bw3_1'])
    conv3 = maxpool2d(conv3)
    conv3 = tf.nn.dropout(conv3, keep_prob)
    conv3_bn = tf.layers.batch_normalization(conv3, name='conv3_bn')
    # print(conv3.get_shape())

    # 16 x 6 x 512
    conv4 = conv2d(conv3_bn, weights['w4_1'], biases['bw4_1'])
    conv4 = maxpool2d(conv4)
    conv4 = tf.nn.dropout(conv4, keep_prob)
    conv4_bn = tf.layers.batch_normalization(conv4, name='conv4_bn')
    # print(conv4.get_shape())

    # 8 x 3 x 1024
    fc1 = tf.contrib.layers.flatten(conv4_bn)
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['fc1']), biases['bfc1']))
    fc1 = tf.nn.dropout(fc1, keep_prob2)
    # print(fc1.get_shape())

    # 1024
    out = tf.add(tf.matmul(fc1, weights['fc2']), biases['bfc2'])
    return out

weights = {
    'w1_1' : tf.Variable(tf.random_normal([3, 3, 1, 64]), name='w1_1'),
    'w2_1' : tf.Variable(tf.random_normal([3, 3, 64, 128]), name='w2_1'),
    'w3_1' : tf.Variable(tf.random_normal([3, 3, 128, 256]), name='w3_1'),
    'w4_1' : tf.Variable(tf.random_normal([3, 3, 256, 512]), name='w4_1'),
    'fc1' : tf.Variable(tf.random_normal([8*3*512, 512]), name='fc1'),
    'fc2' : tf.Variable(tf.random_normal([512, num_classes]), name='fc2')
}

biases = {
    'bw1_1' : tf.Variable(tf.random_normal([64]), name='bw1_1'),
    'bw2_1' : tf.Variable(tf.random_normal([128]), name='bw2_1'),
    'bw3_1' : tf.Variable(tf.random_normal([256]), name='bw3_1'),
    'bw4_1' : tf.Variable(tf.random_normal([512]), name='bw4_1'),
    'bfc1': tf.Variable(tf.random_normal([512]), name='bfc1'),
    'bfc2': tf.Variable(tf.random_normal([num_classes]), name='bfc2')
}

x = tf.placeholder(tf.float32, [None, feature_dim])
keep_prob = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
y_conv = cnn(x, weights, biases, keep_prob, keep_prob2)
y_conv_softmax = tf.nn.softmax(y_conv)


tr = transformMEL(bins=43, frameSize=1024, hopSize=512)
audio, sampleRate, bitrate = util.readAudioScipy(audio_file_name) 
melspec = tr.compute_transform2(audio.sum(axis=1), sampleRate=sampleRate)
num_batch = int(melspec.shape[0]/batch_size)
melspec_tensor = np.zeros((num_batch,feature_dim))

for i in range(0, num_batch):
    if i+batch_size > melspec.shape[0]:
        break
    # plt.imshow(melspec[i*batch_size:(i+1)*batch_size].T,interpolation='none', origin='lower')
    # plt.show()
    melspec_tensor[i, :] = melspec[i*batch_size:(i+1)*batch_size].reshape(-1, feature_dim)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path + '-'+ str(ckpt_load_idx))
    print("Model restored.")

    prediction_result = y_conv_softmax.eval(feed_dict={x: melspec_tensor, keep_prob: 1.0, keep_prob2: 1.0})
    for i in range(0, num_batch):
        print(instruments[np.argmax(prediction_result[i,:])])