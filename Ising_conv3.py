# Thirdparty libraries
import numpy as np
import tensorflow as tf
import data_reader

import os
from clusterone import get_data_path, get_logs_path

sess = tf.InteractiveSession()

PATH_TO_LOCAL_LOGS = os.path.expanduser('~/Desktop/Clusterone/ML_Ising/logs/')
ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser('~/Desktop/Clusterone/ML_Ising')

try:
  job_name = os.environ['JOB_NAME']
  task_index = os.environ['TASK_INDEX']
  ps_hosts = os.environ['PS_HOSTS']
  worker_hosts = os.environ['WORKER_HOSTS']
except:
  job_name = None
  task_index = 0
  ps_hosts = None
  worker_hosts = None

flags = tf.app.flags

# Training related flags
flags.DEFINE_string("data_dir",
                    get_data_path(
                        dataset_name = "kelvinchngphysict/2d10", #all mounted repo
                        local_root = ROOT_PATH_TO_LOCAL_DATA,
                        local_repo = "2d10",
                        path = ''
                        ),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")
flags.DEFINE_string("log_dir",
                     get_logs_path(root=PATH_TO_LOCAL_LOGS),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to clusterone without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")

FLAGS = flags.FLAGS

print(FLAGS.data_dir+"2D10_p_replicas_38_43_50000_%.1d.txt")

HSF = data_reader.insert_file_info(FLAGS.data_dir+"2D10_p_replicas_38_43_50000_%.1d.txt", np.arange(1,2), load_test_data_only=False)
HSF = HSF.categorize_data()

# print(HSF.train.images)

n_x = 10
n_output_neuron = 2

filter_d = 2
filter_h = 2

n_feature_map1 = 16
n_feature_map2 = 8
n_fully_connected_neuron = 16

batch_size = 50
n_train_data = 50000

# Adaptive learning rate is used. As the training goes on, the learning rate is
# lowered progressively using exponential decay function.
#   Optimizer initial learning rate
eta0 = 1e-3

#   decay rate
decay_rate = 0.825

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def bn(inputs, training=True) :
    return tf.contrib.layers.batch_norm(inputs,
                                         center=True,
                                         scale=True,
                                         is_training=training)
                                         
x = tf.placeholder(tf.float32, [None, n_x*n_x])
y_ = tf.placeholder(tf.float32, [None, n_output_neuron])

# Feature extraction layer -----------------------------------------------------------

# First Convolution Layer
W_conv1 = weight_variable([filter_d,filter_h,1,n_feature_map1])
b_conv1 = bias_variable([n_feature_map1]) # 16

x_image = tf.reshape(x, [-1,n_x,n_x,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Second Convolution Layer
W_conv2 = weight_variable([filter_d,filter_h,n_feature_map1,n_feature_map2])
b_conv2 = bias_variable([n_feature_map2])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

# Classification layer ---------------------------------------------------------------

# Fully-connected Layer
W_fc1 = weight_variable([8*8*n_feature_map2, n_fully_connected_neuron]) 
b_fc1 = bias_variable([n_fully_connected_neuron])

h_conv1_flat = tf.reshape(h_conv2, [-1, 8*8*n_feature_map2])
h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([n_fully_connected_neuron,n_output_neuron])
b_fc2 = bias_variable([n_output_neuron]) 

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False)
# Use adaptive learning rate
global_step = tf.Variable(0, trainable=False)
eta = tf.train.exponential_decay(eta0, global_step*batch_size, n_train_data, decay_rate)
train_step = tf.train.AdamOptimizer(eta).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Before Variables can be used within a session, they must be initialized
# using that session.
sess.run(tf.global_variables_initializer())

# Training

for i in range(100): 

    batch = HSF.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

np.savetxt("test.txt",np.array([train_accuracy]))
