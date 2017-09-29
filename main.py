from __future__ import print_function

import os
import time
import math
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.contrib import rnn

import matplotlib
import matplotlib.pyplot as plt

from utils import *
from model import BiGRU

FLAGS = None

def main(_):
    # Hypterparameters
    num_features = 66
    num_classes = 8
    learning_rate = 0.001
    dropout = False
    dropout_rate = 0.7
    clip_gradient = True
    max_grad_norm = 2.0
    state_size = FLAGS.state_size
    print("state_size:", state_size)
    num_layers = 3
    num_outputs_fc1 = 100
    max_seq_len = 400
    batch_size = 64

    # Training parameters
    val_interval = batch_size * 5
    print_interval = batch_size * 10
    samples_to_process = 1e4
    early_stopping = True
    patience = 4
    patience_count = 0

    samples_processed = 0
    loss_training = []
    samples_val = []
    costs, accs_val, grads_norm = [], [], []

    # Data file directories
    relative_path = './data/SetOf7604Proteins/'
    trainList_addr = relative_path + 'trainList'
    validList_addr = relative_path + 'validList'
    testList_addr = relative_path + 'testList'

    rnn = BiGRU(num_features, num_classes, learning_rate, dropout, dropout_rate, clip_gradient,
                max_grad_norm, state_size, num_layers)
    print("Graph build seems successful!")

    start = time.time()
    train_list = read_list(trainList_addr)
    valid_list = read_list(validList_addr)
    test_list = read_list(testList_addr)

    train_generator = generate_batch(train_list, relative_path, max_seq_len, batch_size)

    X_train, y_train, len_train, mask_train = train_generator.next()
    X_valid, y_valid, len_valid, mask_valid = read_data(valid_list, relative_path, max_seq_length=683) # 683

    print("Time spent on loading data: {:.1f}".format(time.time() - start))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Start training!!!")
        start = time.time()
        while samples_processed < samples_to_process:
            loss_train, grad_norm_train = rnn.train_on_batch(sess, X_train, y_train, len_train, mask_train)
            samples_processed += batch_size
            grads_norm += [grad_norm_train]

            #validation data
            if samples_processed % val_interval == 0:
                acc_val, loss_val = rnn.evaluate_on_batch(sess, X_valid, y_valid, len_valid, mask_valid)
                costs += [loss_train]
                samples_val += [samples_processed]
                accs_val += [acc_val]

                if samples_processed % print_interval == 0:
                    print("samples_processed: %d, batch_cost: %.3f, validation_accs: %.4f, validation_loss: %.4f, patience_count: %d" % \
                          (samples_processed, loss_train, acc_val, loss_val, patience_count))

                if early_stopping:
                    if len(accs_val) > patience and acc_val < accs_val[-2]:
                        patience_count += 1
                    if patience_count >= patience:
                        break
        print("time spent: {:.3f} seconds".format(time.time() - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a RNN model to predict protein secondary structure prediction")
    parser.add_argument('-s', '--state_size', type=int, default=100, help="State size of RNN cell")
    # parser.add_argument()
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)