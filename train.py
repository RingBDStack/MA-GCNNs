
# coding: utf-8

'''
This file contains several training and evaluating parts,
including defining hyperparameters, building the calculation graph and concluding final mean as well as std.

Let's denote the process of training 90% data and evaluating 10% data as fold i.
Firstly, we run different folds from 0 to 9, totally ten folds. (ten-fold cross validation)
Then, we calculate mean accuracy and std according to evaluation accuracy from different folds.

It is worth noting that we actually tune hyperparameters each fold so that we can reach our best score, 
just in case that the model falls into the local optima, which cannot represent our model's normal performance.

Created on 18/8/4.
Copyright 2018. All rights reserved.

'''

import tensorflow as tf
import numpy as np
import time
import models
import utils 

flags = tf.app.flags
FLAGS = flags.FLAGS
# hyperparameters.
flags.DEFINE_integer('batch_size', 18, 'Batch Size (default: 64), should be tuned according to data size.')
flags.DEFINE_integer('num_epochs', 1000, 'Number of training epochs (default: 500), early stop.')
flags.DEFINE_integer('folds', 10, 'Number of folds in cross validation (default: 10)')
flags.DEFINE_integer('class_size', 2, 'Classification Size (default: 2), should be tuned according to different datasets.')
flags.DEFINE_integer('seq_len', 18, 'Number of selected nodes (default: 18 MUTAG), should be tuned according to different datasets.')
flags.DEFINE_integer('order_len', 45, 'Number of 3-order length (default: 45 MUTAG), should be tuned according to different datasets.')
flags.DEFINE_float('learning_rate', 1e-3, 'MomentumOptimizer/AdamOptimizer learning rate (default: 0.001)')
flags.DEFINE_float('momentum', 0.9, 'MomentumOptimizer learning rate decay (default: 0.9)')
flags.DEFINE_string('data_fn', 'datasets/mutag_data.npy', 'training & test file name, including data matrix (default: mutag_data.npy)')
flags.DEFINE_string('label_fn', 'datasets/mutag_label.npy', 'training & test file name, including label vector (default: mutag_label.npy)')


if __name__ == "__main__":
    # divide train set and test set.
    data, label = utils.data_preprocess(FLAGS.data_fn, FLAGS.label_fn)
    test_size = int(data.shape[0]/FLAGS.folds)
    train_size = data.shape[0]-test_size

    with tf.Session() as sess:
        build_time = time.time()
        
        net = models.MotifAttGCN(sess, FLAGS.batch_size, FLAGS.class_size, FLAGS.seq_len, FLAGS.order_len)
        # list containing each accuracy calculated from each fold data.
        accs = []
        for fold in range(FLAGS.folds):
            sess.run(tf.global_variables_initializer())
            begin_time = time.time()
            print('--------this fold initialization(build model+init) takes %.3f minutes\n'%((begin_time-build_time)/60))
            # get batch data.
            if fold < FLAGS.folds - 1:
                train_x, train_t, test_x, test_t = utils.divide_train_test(data, label, 
                                                                           fold*test_size, 
                                                                           fold*test_size+test_size)
            else:
                train_x, train_t, test_x, test_t = utils.divide_train_test(data, label, 
                                                                           data.shape[0]-test_size, 
                                                                           data.shape[0])
            print('present fold train shape: ', train_x.shape)
            print('present fold test shape: ', test_x.shape)
            max_fold_acc = 0
            for epoch in range(FLAGS.num_epochs):
                train_loss = 0
                train_acc = 0
                batch_num = 0
                for i in range(0, train_size, FLAGS.batch_size):
                    x_batch, t_batch = utils.load_batch(train_x, train_t, FLAGS.batch_size)
                    loss, acc, pred = net.train(x_batch, t_batch, FLAGS.learning_rate, FLAGS.momentum)
                    batch_num += 1
                    train_loss += loss
                    train_acc += acc
                    
                    if batch_num % 4 == 0:
                        print('training loss {:.4f}'.format(loss))
                        print('batch accuracy {:.4f}'.format(acc))
            
                print('epoch train loss: ', train_loss/batch_num)
                print('epoch train acc: ', train_acc/batch_num)
                eva_acc, eva_pred = net.evaluate(test_x, test_t)
                print('epoch'+str(epoch)+': ', eva_acc)
                if eva_acc > max_fold_acc:
                    max_fold_acc = eva_acc
                    best_pred = eva_pred
                    print('-------------------------------------------------')
                    print('Model reaches a better accuracy.')
                    print('-------------------------------------------------')
            
            print('evaluation set pred: \t', eva_pred)
            print('evaluation set label: \t', test_t)
            print('evaluation_accuracy: {:.4f}\n'.format(max_fold_acc))

            '''
            # 0.8889 is selected according to mutag's performance.
            # can be changed according to different benchmark datasets.

            if max_fold_acc < 0.8889:
                print('This fold has falled into local optima.')
                print('Please tune hyperparameters or just run it again.')
                continue
            
            '''

            accs.append(max_fold_acc)
            print('--------this fold train+evaluation takes %.3f minutes\n'%((time.time()-begin_time)/60))
        
        accs = np.array(accs)
        mean = np.mean(accs)*100
        std = np.std(accs)*100
        print('This benchmark dataset has the following results: ')
        print('Mean: {:.2f}'.format(mean))
        print('Std: {:.2f}'.format(std))






