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
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Data file directories
relative_path = '../data/SetOf7604Proteins/'
trainList_addr = relative_path + 'trainList'
validList_addr = relative_path + 'validList'
testList_addr = relative_path + 'testList'

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
    num_layers = 3
    num_outputs_fc1 = 100
    max_seq_len = 400
    batch_size = 64

    # Training parameters
    epochs = 3
    valid_interval = 5
    losses_train, grads_norm = [], []
    losses_valid, accuracies = [], []

    rnn = BiGRU(num_features, num_classes, learning_rate, dropout, dropout_rate, clip_gradient,
                max_grad_norm, state_size, num_layers, batch_size)
    print("Graph build seems successful!")

    train_list = read_list(trainList_addr)
    valid_list = read_list(validList_addr)
    test_list = read_list(testList_addr)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Start training!!!")
        start = time.time()
        for epoch in xrange(epochs):
            print("Epoch {} out of {}".format(epoch + 1, epochs))
            for i, batch in enumerate(minibatches(train_list, relative_path, batch_size, max_seq_len)):
                loss_train, grad_norm = rnn.train_on_batch(sess, *batch)
                losses_train.append(loss_train)
                grads_norm.append(grad_norm)
                if i % valid_interval == 0:
                    print("Current batch loss {:.3f}".format(loss_train))
                    # loss_valid, accuracy = rnn.evaluate(sess, valid_list, relative_path)
                    # losses_valid.append(loss_valid)
                    # accuracies.append(accuracy)
                    # print("samples_processed: %d, batch_cost: %.3f, validation_accs: %.4f, validation_loss: %.4f" % \
                    #       (i * batch_size, loss_train, accuracy, loss_valid))

        print("time spent: {:.3f} seconds".format(time.time() - start))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a RNN model to predict protein secondary structure prediction")
    parser.add_argument('-s', '--state_size', type=int, default=100, help="State size of RNN cell")
    # parser.add_argument()
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)