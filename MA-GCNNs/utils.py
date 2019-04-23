
# coding: utf-8

'''
This file contains some functions used to preprocess data.
Our data are numpy files, and we use ten-fold cross validation.

These functions are used in train.py

Created on 18/8/4.
Copyright 2018. All rights reserved.

'''

import numpy as np

def data_preprocess(data_fn, label_fn):
    data = np.load(data_fn)
    label = np.load(label_fn)
    print(data.shape)
    print(label.shape)
    return data, label

def divide_train_test(data, label, test_begin_idx, test_end_idx):
    data_size = data.shape[0]
    train_x = np.concatenate([data[0:test_begin_idx], data[test_end_idx:data_size]])
    train_t = np.concatenate([label[0:test_begin_idx], label[test_end_idx:data_size]])
    
    test_x = data[test_begin_idx:test_end_idx]
    test_t = label[test_begin_idx:test_end_idx]
    
    return train_x, train_t, test_x, test_t

def load_batch(x, t, batch_size):
    index = np.random.randint(x.shape[0], size = batch_size)
    batch = (x[index], t[index])
    return batch[0], batch[1]


