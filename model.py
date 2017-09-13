from __future__ import print_function
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.contrib.layers import fully_connected, batch_norm

from utils import *
import custom_ops



# ==========
#   MODEL
# ==========

# Defining hyperparameters
num_iterations = 3e4
batch_size=64
number_inputs=66
number_outputs=8
seq_len=30 # max 700
learning_rate = 0.001
test = True

# Get all the data
trainList_addr = './data/trainList'
validList_addr = './data/validList'
testList_addr = './data/testList'

train_list, train_len_list = read_list(trainList_addr)
valid_list, valid_len_list = read_list(validList_addr)
test_list, test_len_list = read_list(testList_addr)

train_generator = generate_batch(train_list, train_len_list,
                                 max_seq_length=seq_len,
                                 batch_size=batch_size)

X_train, t_train, len_train = train_generator.next()
X_valid, t_valid, len_valid = read_data(valid_list, valid_len_list,
                                        max_seq_length=seq_len)
X_test, t_test, len_test = read_data(test_list, test_len_list,
                                     max_seq_length=seq_len)
if test:
    print("X_train: ", type(X_valid), X_valid.shape)
    print("X_length: ", type(len_valid), len_valid.shape)
    print("t_train: ", type(t_valid), t_valid.shape)

# Defining the graph
reset_default_graph()

X_input = tf.placeholder(tf.float32, shape=[None, seq_len, number_inputs], name='X_input')
X_length = tf.placeholder(tf.int32, shape=[None,], name='X_length')
t_input = tf.placeholder(tf.int32, shape=[None, seq_len], name='t_input')

num_units_encoder = 100
num_units_l1 = 100

cell_fw = tf.nn.rnn_cell.GRUCell(num_units_encoder)
cell_bw = tf.nn.rnn_cell.GRUCell(num_units_encoder)
#enc_cell = tf.nn.rnn_cell.OutputProjectionWrapper(enc_cell, number_outputs)
enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=X_input,
                                                 sequence_length=X_length, dtype=tf.float32)
enc_outputs = tf.concat(enc_outputs, 2)
print("enc_outputs shape is ", enc_outputs.get_shape())
outputs = tf.reshape(enc_outputs, [-1, num_units_encoder*2])
l1 = fully_connected(outputs, num_units_l1, normalizer_fn=batch_norm)
l_out = fully_connected(l1, number_outputs, activation_fn=None)
print("l_out shape is ", l_out.get_shape())

batch_size_shp = tf.shape(enc_outputs)[0]
seq_len_shp = tf.shape(enc_outputs)[1]
l_out_reshape = tf.reshape(l_out, [batch_size_shp, seq_len_shp, number_outputs])

y = l_out_reshape

def loss_and_acc(preds):
    # sequence_loss_tensor is a modification of TensorFlow's own sequence_to_sequence_loss
    # TensorFlow's seq2seq loss works with a 2D list instead of a 3D tensors
    loss = custom_ops.sequence_loss(preds, t_input)
    # if you want regularization
    #reg_scale = 0.00001
    #regularize = tf.contrib.layers.l2_regularizer(reg_scale)
    #params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #reg_term = sum([regularize(param) for param in params])
    #loss += reg_term
    # calculate accuracy
    argmax = tf.to_int32(tf.argmax(preds, 2))
    correct = tf.to_float(tf.equal(argmax, t_input))
    accuracy = tf.reduce_sum(correct) / tf.reduce_prod(tf.cast(tf.shape(t_input), tf.float32))
    return loss, accuracy, argmax

loss, accuracy, predictions = loss_and_acc(y)

# use lobal step to keep track of our iterations
global_step = tf.Variable(0, name='global_step', trainable=False)
# pick optimizer, try momentum or adadelta
optimizer = tf.train.AdamOptimizer(learning_rate)
# extract gradients for each variable
grads_and_vars = optimizer.compute_gradients(loss)

# add below for clipping by norm
#gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
#clipped_gradients, global_norm = (
#    tf.clip_by_global_norm(gradients, self.clip_norm) )
#grads_and_vars = zip(clipped_gradients, variables)

# apply gradients and make trainable function
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# restricting memory usage, TensorFlow is greedy and will use all memory otherwise
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# initialize the Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
sess.run(tf.global_variables_initializer())

if test:
    print("=" * 10 + "validating the model"+ "=" * 10)
    # test validation part
    sess.run(tf.global_variables_initializer())
    feed_dict = {X_input: X_valid, X_length: len_valid, t_input: t_valid}
    fetches = [y]
    res = sess.run(fetches=fetches, feed_dict=feed_dict)
    print("y", res[0].shape)
    print("=" * 10 + "Model validation finished"+ "=" * 10)

setting up running parameters
val_interval = batch_size*2
samples_to_process = 5000
samples_processed = 0
samples_val = []
costs, accs_val = [], []

while samples_processed < samples_to_process:
    # make fetches
    fetches_tr = [train_op, loss, accuracy]
    # set up feed dict
    feed_dict_tr = {X_input: X_train, X_length: len_train,
                    t_input: t_train}
    # run the model
    res = tuple(sess.run(fetches=fetches_tr, feed_dict=feed_dict_tr))
    _, batch_cost, batch_acc = res

    samples_processed += batch_size
    print("samples_processed: {}, batch_cost: {}, batch_acc: {}".format(samples_processed, batch_cost, batch_acc))
    #validation data
    if samples_processed % val_interval == 0:
        print("validating")
        fetches_val = [accuracy, y]
        feed_dict_val = {X_input: X_valid, X_length: len_valid,
                         t_input: t_valid}
        res = tuple(sess.run(fetches=fetches_val, feed_dict=feed_dict_val))
        acc_val, output_val = res
        costs += [batch_cost]
        samples_val += [samples_processed]
        accs_val += [acc_val]
        print("validation_accs", acc_val)

fig, ax1 = plt.subplots()
plt.plot(samples_val, accs_val, 'b-')
ax1.set_ylabel('Validation Accuracy', fontsize=15)
ax1.set_xlabel('Processed samples', fontsize=15)
plt.title('', fontsize=20)
ax2 = ax1.twinx()
ax2.plot(samples_val, costs, 'r-')
ax2.set_ylabel('Training Cost', fontsize=15)
plt.grid('on')
plt.savefig("out.png")
plt.show()