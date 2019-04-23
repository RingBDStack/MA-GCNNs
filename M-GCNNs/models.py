
# coding: utf-8

'''
This file contains one model, our baseline: M-GCNNs.
M-GCNNs:
    convolution layer + convolution layer + fully connected layer + fully connected layer + fully connected layer

Created on 18/8/4.
Copyright 2018. All rights reserved.

'''

import tensorflow as tf
import layers

class MGCNNs(object):
    def __init__(self, session, 
                 batch_size,
                 class_size,
                 seq_len,
                 order_len,
                 conv1_ksize=3, 
                 conv1_output_channels=32,
                 conv2_ksize=3, 
                 conv2_output_channels=64,
                 dropout_ratio=0.5,
                 fc1_output_size=128,
                 fc2_output_size=64,
                 k = 3):
        self.batch_size = batch_size
        self.class_size = class_size
        self.seq_len = seq_len
        self.order_len = order_len
        self.conv1_ksize = conv1_ksize
        self.conv1_output_channels = conv1_output_channels
        self.conv2_ksize = conv2_ksize
        self.conv2_output_channels = conv2_output_channels
        self.dropout_ratio = dropout_ratio
        self.fc1_output_size = fc1_output_size
        self.fc2_output_size = fc2_output_size
        self.k = k
        
        self.build_placeholders()
        
        self.loss, self.probabilities = self.forward_propagation()
        self.pred = tf.to_int32(tf.argmax(self.probabilities, 1))
        correct_prediction = tf.equal(self.pred, self.t)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print('Forward propagation finished.')
        
        self.sess = session
        self.optimizer = tf.train.MomentumOptimizer(self.lr, self.mom).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables())
        print('Backward propagation finished.')
        
    def build_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, self.order_len, self.seq_len*self.k], 'graph_nodes')
        self.t = tf.placeholder(tf.int32, [None], 'labels')
        self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
        self.mom = tf.placeholder(tf.float32, [], 'momentum')
        
    def forward_propagation(self):
        with tf.variable_scope('conv_n1'):
            x = tf.expand_dims(self.x, -1)
            conv1 = tf.layers.conv2d(
                inputs=x,
                filters=self.conv1_output_channels,
                kernel_size=[1, self.conv1_ksize],
                strides=(1, self.k),
                padding="valid",
                activation=tf.nn.relu)

            dropout1 = tf.layers.dropout(
                inputs=conv1, rate=self.dropout_ratio)
    
        with tf.variable_scope('conv_n2'):
            conv2 = tf.layers.conv2d(
                inputs=dropout1,
                filters=self.conv2_output_channels,
                kernel_size=[self.conv2_ksize, 1],
                strides=(3, 1),
                padding="valid",
                activation=tf.nn.relu)

            dropout2 = tf.layers.dropout(
                inputs=conv2, rate=self.dropout_ratio)

        # (batch_size, feature_len, seq_len, filters)
        with tf.variable_scope('fc_n'):
            flat = tf.reshape(dropout2, [-1, dropout2.shape[1]*dropout2.shape[2]*dropout2.shape[3]])
            dense1 = tf.layers.dense(inputs=flat, units=self.fc1_output_size, activation=tf.nn.relu)
            dense2 = tf.layers.dense(inputs=dense1, units=self.fc2_output_size, activation=tf.nn.relu)
            dense3 = tf.layers.dense(inputs=dense2, units=self.class_size, activation=None)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.t, logits=dense3)
        
        return loss, tf.nn.softmax(dense3, name="probabilities")
    
    def train(self, batch_x, batch_t, learning_rate = 1e-3, momentum = 0.9):
        feed_dict = {
            self.x : batch_x,
            self.t : batch_t, 
            self.lr: learning_rate,
            self.mom: momentum
        }
        _, loss, acc, pred = self.sess.run([self.optimizer, self.loss, self.accuracy, self.pred], feed_dict = feed_dict)
        
        return loss, acc, pred
    
    def evaluate(self, batch_x, batch_t):
        feed_dict = {
            self.x : batch_x,
            self.t : batch_t
        }
        acc, pred = self.sess.run([self.accuracy, self.pred], feed_dict = feed_dict)
        
        return acc, pred



