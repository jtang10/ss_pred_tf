from __future__ import print_function

import os
import time
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.contrib import rnn

class BiGRU(object):
    def __init__(self, num_features, num_classes, learning_rate, dropout,
                 dropout_rate, clip_gradient, max_grad_norm, state_size,
                 num_layers):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.clip_gradient = clip_gradient
        self.max_grad_norm = max_grad_norm
        self.state_size = state_size
        self.num_layers = num_layers

        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.lengths_placeholder = None
        self.masks_placeholder = None

        self.build()

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss, self.accuracy = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=[None, None, self.num_features], name='inputs_placeholder')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, None], name='labels_placeholder')
        self.lengths_placeholder = tf.placeholder(tf.int32, shape=[None,], name='lengths_placeholder')
        self.masks_placeholder = tf.placeholder(tf.float32, shape=[None, None], name='masks_placeholder')
        self.phase = tf.placeholder(tf.bool, name='phase')

    def create_feed_dict(self, inputs_batch, labels_batch, lengths_batch, masks_batch, phase=True):
        feed_dict = {self.inputs_placeholder: inputs_batch,
                     self.labels_placeholder: labels_batch,
                     self.lengths_placeholder: lengths_batch,
                     self.masks_placeholder: masks_batch,
                     self.phase: phase}
        return feed_dict

    def GRUCell(self, dropout, state_size):
        if dropout:
            return rnn.DropoutWrapper(rnn.GRUCell(state_size),
                                  output_keep_prob=dropout_keep_rate)
        else:
            return rnn.GRUCell(state_size)

    def add_prediction_op(self):
        cells = rnn.MultiRNNCell([self.GRUCell(self.dropout, self.state_size)
                                 for _ in range(self.num_layers)])
        output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cells, cell_bw=cells,
                                                        inputs=self.inputs_placeholder,
                                                        sequence_length=self.lengths_placeholder,
                                                        dtype=tf.float32)
        outputs = tf.concat(output, 2)
        batch_size_shp = tf.shape(outputs)[0]
        seq_len_shp = tf.shape(outputs)[1]
        outputs = tf.reshape(outputs, [-1, self.state_size * 2])
        fc1 = fully_connected(outputs, self.state_size) # , normalizer=batch_norm
        bn1 = batch_norm(fc1, center=True, scale=True, is_training=self.phase)
        fc2 = fully_connected(bn1, self.num_classes, activation_fn=None)
        pred = tf.reshape(fc2, [batch_size_shp, seq_len_shp, self.num_classes])
        return pred

    def add_loss_op(self, pred):
        mask_sum = tf.reduce_sum(self.masks_placeholder)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder,
                                                                       logits=pred)
        cross_entropy *= self.masks_placeholder
        loss = tf.reduce_sum(cross_entropy) / mask_sum

        labels_pred = tf.argmax(pred, 2, output_type=tf.int32)
        correct_pred = tf.to_float(tf.equal(labels_pred, self.labels_placeholder))
        accuracy = tf.reduce_sum(correct_pred * self.masks_placeholder) / mask_sum

        return loss, accuracy

    def add_training_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads = [element[0] for element in grads_and_vars]
        variables =  [element[1] for element in grads_and_vars]
        if self.clip_gradient:
            grads = tf.clip_by_global_norm(grads, self.max_grad_norm)[0]
            
        self.grad_norm = tf.global_norm(grads)
        grads_and_vars = [(grads[i], variables[i]) for i in range(len(grads))]
        # apply gradients and make trainable function
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, lengths_batch, masks_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch, lengths_batch, masks_batch)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def evaluate_on_batch(self, sess, inputs_batch, labels_batch, lengths_batch, masks_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch, lengths_batch, masks_batch, phase=False)
        accuracy, loss = sess.run([self.accuracy, self.loss], feed_dict=feed)
        return accuracy, loss

    def train_and_evaluate(self, sess, train, validation, valild_interval=5):
        pass






