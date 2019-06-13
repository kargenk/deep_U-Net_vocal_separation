#!usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import const_for_colab as C
import util_for_colab as util
import os

# tf.nnのfilter = [kernel_height, kernel_width, input_channel, output_channel]
# 畳み込み層(エンコーダ部分)
def conv2d(x, output_channel, kernel=5, stride=2, pad='same', batch_norm=True, is_train=True, leaky_relu=True):
    net = tf.layers.conv2d(x,
                           filters=output_channel,
                           kernel_size=[kernel, kernel],
                           strides=[stride, stride],
                           padding=pad)
    if leaky_relu:
        net = tf.nn.leaky_relu(net, 0.2)
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    return net

# tf.nnのfilter = [kernel_height, kernel_width, output_channel, input_channel]
# output_shape = [バッチ数, 得たいheight, 得たいwidth, 得たいchannel]
# 逆畳み込み層(デコーダ部分)
def de_conv2d(x, output_channel, kernel=5, stride=2, pad='same', batch_norm=True, is_train=True, relu=True):
    net = tf.layers.conv2d_transpose(x,
                                     filters=output_channel,
                                     kernel_size=[kernel, kernel],
                                     strides=[stride, stride],
                                     padding=pad)
    if relu:
        net = tf.nn.relu(net)
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    return net

# ミニバッチ毎に処理する関数(ジェネレータを返す)
def batch_generator(X, y, batch_size=C.BATCH_SIZE, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
        
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size]) # 段階的に返す

class UNet(object):
    def __init__(self, batchsize=C.BATCH_SIZE, epochs=30, learning_rate=1e-4,
                 dropout_rate=0.5, shuffle=True, random_seed=None):
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.global_variables_initializer()  # 変数を初期化
            self.saver = tf.train.Saver()
        
        # セッションを作成し，計算グラフgを渡す
        self.sess = tf.Session(graph=g)
    
    def build(self):
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
        tf_x = tf.placeholder(tf.float32, shape=[None, 512, 128, 1], name='tf_x')
        tf_y = tf.placeholder(tf.float32, shape=[None, 512, 128, 1], name='tf_y')
        conv1 = conv2d(tf_x, output_channel=16)
        conv2 = conv2d(conv1, output_channel=32)
        conv3 = conv2d(conv2, output_channel=64)
        conv4 = conv2d(conv3, output_channel=128)
        conv5 = conv2d(conv4, output_channel=256)
        conv6 = conv2d(conv5, output_channel=512)
        de_conv1 = de_conv2d(conv6, output_channel=256)
        concat1 = tf.concat([de_conv1, conv5], axis=-1)
        dropout1 = tf.layers.dropout(concat1, rate=self.dropout_rate, training=is_train)
        de_conv2 = de_conv2d(dropout1, output_channel=128)
        concat2 = tf.concat([de_conv2, conv4], axis=-1)
        dropout2 = tf.layers.dropout(concat2, rate=self.dropout_rate, training=is_train)
        de_conv3 = de_conv2d(dropout2, output_channel=64)
        concat3 = tf.concat([de_conv3, conv3], axis=-1)
        dropout3 = tf.layers.dropout(concat3, rate=self.dropout_rate, training=is_train)
        de_conv4 = de_conv2d(dropout3, output_channel=32)
        concat4 = tf.concat([de_conv4, conv2], axis=-1)
        de_conv5 = de_conv2d(concat4, output_channel=16)
        concat5 = tf.concat([de_conv5, conv1], axis=-1)
        de_conv6 = de_conv2d(concat5, output_channel=1, relu=False)
        activation_final = tf.math.sigmoid(de_conv6)
        
        # 損失関数
        pred = tf_x * activation_final  # マスクをかけた元音源
        loss = tf.reduce_mean(
                tf.map_fn(tf.abs, (pred - tf_y)),
                name='mean_absolute_error')
        # 最適化
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(loss, name='train_op')
        
    def save(self, epoch, path='./tf-layers-model/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        print('Saving model in %s' % path)
        self.saver.save(self.sess,
                        os.path.join(path, 'model.ckpt'),
                        global_step=epoch)

    def load(self, epoch, path):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess,
                           os.path.join(path, 'model.ckpt-%d' % epoch))

    def train(self, X_list, y_list, initialize=True, subepoch=None):
        # 変数を初期化
        if initialize:
            self.sess.run(self.init_op)

        self.train_cost_ = []
#         X_data = np.array(training_set[0])
#         y_data = np.array(training_Set[1])
        item_count = len(X_list)
        item_length = [x.shape[1] for x in X_list]
        subepoch = sum(item_length) // C.PATCH_LENGTH // C.BATCH_SIZE * 4

        for epoch in range(1, self.epochs + 1):
#             batch_gen = batch_generator(X_data, y_data, shuffle=self.shuffle)
            avg_loss = 0.0

            for i in range(subepoch):
                print('subepoch: ' + str(i))
                X = np.zeros((C.BATCH_SIZE, 512, C.PATCH_LENGTH, 1),
                              dtype="float32")
                Y = np.zeros((C.BATCH_SIZE, 512, C.PATCH_LENGTH, 1),
                              dtype="float32")
                idx_item = np.random.randint(0, item_count, C.BATCH_SIZE)
                for i in range(C.BATCH_SIZE):
                    randidx = np.random.randint(
                        item_length[idx_item[i]]-C.PATCH_LENGTH-1)
                    X[i, :, :, 0] =                         X_list[idx_item[i]][1:, randidx:randidx+C.PATCH_LENGTH]
                    Y[i, :, :, 0] =                         y_list[idx_item[i]][1:, randidx:randidx+C.PATCH_LENGTH]
                    feed = {'tf_x:0': X,
                            'tf_y:0': Y,
                            'is_train:0': True} # ドロップアウト
#                 for k, (batch_x, batch_y) in enumerate(batch_gen):
#                     feed = {'tf_x:0': batch_x,
#                             'tf_y:0': batch_y,
#                             'is_train:0': True} # ドロップアウト
                    loss, _ = self.sess.run(['mean_absolute_error:0', 'train_op'],
                                            feed_dict=feed)
                    avg_loss += loss

            print('Epoch %2d: Training Avg. Loss: %7.3f' % (epoch, avg_loss))
            model.save(epoch=epoch)