# Thirdparty libraries
import numpy as np
import tensorflow as tf
import data_reader

from scipy import signal
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
                        dataset_name = "kelvinchngphysicist/2d10", #all mounted repo
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

n_x = 10

Ising = data_reader.insert_file_info(FLAGS.data_dir+"2D%d"%(n_x)+"_p_%.1d.txt", np.arange(1,3), load_test_data_only=False)
Ising = Ising.categorize_data()

n_output_neuron = 2

filter_d = 2
filter_h = 2

n_feature_map1 = 16
n_feature_map2 = 8
n_fully_connected_neuron = 16

batch_size = 50
n_train_data = 50000

n_steps = 8000
save_freq = 200
n_save = n_steps/save_freq
n_train_data = len(Ising.train.labels)
n_epochs = (float(n_steps)*batch_size)/n_train_data
epochs = np.linspace(0,n_epochs,n_save)

eta_policy = 'cyclical_traig' # 'test'
etaMax = 0.00035
eta0 = 1e-6
n_train_iter = int(float(n_train_data)*n_epochs/batch_size)

if eta_policy in ['test'] :
    eta_lower_bound = max(1e-16,eta0)
    eta0 = eta_lower_bound
    print('Learning rate', '%s - %s'  %(eta_lower_bound, etaMax))
    eta = ((eta_lower_bound/etaMax)**np.linspace(1,0,n_train_iter))*etaMax

if eta_policy in ['cyclical_traig', 'Cyclial triangular'] :
    eta0 = etaMax/4.
    n_train_iter_per_epoch =n_train_iter/n_epochs

    if n_epochs > 9 : 
        cycle_multiplier=min(2,n_epochs)
    else :
        cycle_multiplier=1

    eta_tmp = signal.triang(n_train_iter_per_epoch*cycle_multiplier)*(etaMax-eta0)+eta0

    if n_epochs > 9 :
        cycle = 10
    else :
        cycle = int(n_epochs/cycle_multiplier)

    eta = np.ones(n_train_iter)*eta0
    for i in range(cycle) :
        start = int((i*cycle_multiplier)*n_train_iter_per_epoch)
        end   = int(((i+1)*cycle_multiplier)*n_train_iter_per_epoch)
        eta[start:end] = eta_tmp

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
b_conv1 = bias_variable([n_feature_map1])

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

learning_rate = tf.placeholder(tf.float32,shape=[])

# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False)
# Use adaptive learning rate
#global_step = tf.Variable(0, trainable=False)
# eta = tf.train.exponential_decay(eta0, global_step*batch_size, n_train_data, decay_rate)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Before Variables can be used within a session, they must be initialized
# using that session.
sess.run(tf.global_variables_initializer())

# Training

training_table = np.zeros((n_save,4))
training_table[:,0] = epochs

n = 0
for i in range(n_steps):

    batch = Ising.train.next_batch(50)
    if eta_policy in ['test'] :
      if i % save_freq == 0:
        test_accuracy = accuracy.eval(feed_dict={
            x: Ising.test.images, y_: Ising.test.labels, keep_prob: 1.0})
        training_table[n,0], training_table[n,1] = eta[i], test_accuracy
        print('step %d, eta %g, testing accuracy %g' % (i, eta[i], test_accuracy))
        n+=1
    else :
      if i % save_freq == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: Ising.train.images, y_: Ising.train.labels, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={
            x: Ising.test.images, y_: Ising.test.labels, keep_prob: 1.0})
        training_table[n,1], training_table[n,2], training_table[n,3] = eta[i], train_accuracy, test_accuracy
        print('step %d, training accuracy %g, testing accuracy %g' % (i, train_accuracy, test_accuracy))
        n+=1
    train_step.run(feed_dict={x: batch[0], y_: batch[1], learning_rate: eta[i], keep_prob: 0.5})

if eta_policy in ['test'] :
    np.savetxt(FLAGS.log_dir+"/Ising2D10_LRtest_eta%.3f_%.3f.txt"%(eta0,etaMax),training_table[:,:2])
else :
    np.savetxt(FLAGS.log_dir+"/Ising2D10_CLR_eta%.3f_%.3f.txt"%(eta0,etaMax),training_table)

if eta_policy not in ['test'] :
    n_test_data = len(Ising.test.labels)

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: Ising.test.images, y_: Ising.test.labels, keep_prob: 1.0}))

    n_temp = len(np.unique(Ising.test.temps))
    table = np.zeros((n_temp,2))
    table[:,0] = np.unique(Ising.test.temps)

    for i in range(n_test_data) :
        table[Ising.test.temps[i]==table[:,0],1] += np.argmax(y_conv.eval(feed_dict={x: Ising.test.images[i,:].reshape(1,100), keep_prob: 1.0}))

    np.savetxt(FLAGS.log_dir+"/Ising2D10_output.txt",table)
