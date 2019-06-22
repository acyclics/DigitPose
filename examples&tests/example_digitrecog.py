import tensorflow as tf
import numpy as np
import keras.preprocessing.image
import sklearn.model_selection
import keras
import pandas as pd
import datetime
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def normalize_data(data): 
    data = data / data.max() # convert from [0:255] to [0.:1.]
    return data

def dense_to_one_hot(labels_dense, num_classes):
    # convert class labels from scalars to one-hot vectors e.g. 1 => [0 1 0 0 0 0 0 0 0 0]
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def one_hot_to_dense(labels_one_hot):
    # convert one-hot encodings into labels
    return np.argmax(labels_one_hot,1)

def accuracy_from_dense_labels(y_target, y_pred):
    # compute the accuracy of label predictions
    y_target = y_target.reshape(-1,)
    y_pred = y_pred.reshape(-1,)
    return np.mean(y_target == y_pred)

def accuracy_from_one_hot_labels(y_target, y_pred):
    # compute the accuracy of one-hot encoded predictions
    y_target = one_hot_to_dense(y_target).reshape(-1,)
    y_pred = one_hot_to_dense(y_pred).reshape(-1,)
    return np.mean(y_target == y_pred)

class simpleDigit():
    def __init__(self, nn_name):
        self.s_f_conv1 = 3; # filter size of first convolution layer (default = 3)
        self.n_f_conv1 = 36; # number of features of first convolution layer (default = 36)
        self.s_f_conv2 = 3; # filter size of second convolution layer (default = 3)
        self.n_f_conv2 = 36; # number of features of second convolution layer (default = 36)
        self.s_f_conv3 = 3; # filter size of third convolution layer (default = 3)
        self.n_f_conv3 = 36; # number of features of third convolution layer (default = 36)
        self.n_n_fc1 = 576; # number of neurons of first fully connected layer (default = 576)

        self.mb_size = 50 # mini batch size
        self.keep_prob = 0.33 # keeping probability with dropout regularization 
        self.learn_rate_array = [10*1e-4, 7.5*1e-4, 5*1e-4, 2.5*1e-4, 1*1e-4, 1*1e-4,
                                 1*1e-4,0.75*1e-4, 0.5*1e-4, 0.25*1e-4, 0.1*1e-4, 
                                 0.1*1e-4, 0.075*1e-4,0.050*1e-4, 0.025*1e-4, 0.01*1e-4, 
                                 0.0075*1e-4, 0.0050*1e-4,0.0025*1e-4,0.001*1e-4]
        self.learn_rate_step_size = 3 # in terms of epochs

        self.learn_rate = self.learn_rate_array[0]
        self.learn_rate_pos = 0 # current position pointing to current learning rate
        self.index_in_epoch = 0 
        self.current_epoch = 0
        self.log_step = 0.2 # log results in terms of epochs
        self.n_log_step = 0 # counting current number of mini batches trained on
        self.use_tb_summary = False # True = use tensorboard visualization
        self.use_tf_saver = False # True = use saver to save the model
        self.nn_name = nn_name # name of the neural network

        self.perm_array = np.array([])

    def next_mini_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.mb_size
        self.current_epoch += self.mb_size/len(self.x_train)  
        # adapt length of permutation array
        if not len(self.perm_array) == len(self.x_train):
            self.perm_array = np.arange(len(self.x_train))
        # shuffle once at the start of epoch
        if start == 0:
            np.random.shuffle(self.perm_array)
        # at the end of the epoch
        if self.index_in_epoch > self.x_train.shape[0]:
            np.random.shuffle(self.perm_array) # shuffle data
            start = 0 # start next epoch
            self.index_in_epoch = self.mb_size # set index to mini batch size
            if self.train_on_augmented_data:
                # use augmented data for the next epoch
                self.x_train_aug = normalize_data(self.generate_images(self.x_train))
                self.y_train_aug = self.y_train
        end = self.index_in_epoch
        if self.train_on_augmented_data:
            # use augmented data
            x_tr = self.x_train_aug[self.perm_array[start:end]]
            y_tr = self.y_train_aug[self.perm_array[start:end]]
        else:
            # use original data
            x_tr = self.x_train[self.perm_array[start:end]]
            y_tr = self.y_train[self.perm_array[start:end]]
        return x_tr, y_tr

    def weight_variable(self, shape, name = None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name = name)

    def bias_variable(self, shape, name = None):
        # bias initialization
        initial = tf.constant(0.1, shape=shape) #  positive bias
        return tf.Variable(initial, name = name)

    def conv2d(self, x, W, name = None):
        # 2D convolution
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)

    def max_pool_2x2(self, x, name = None):
        # max pooling
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name = name)

    def summary_variable(self, var, var_name):
        # attach summaries to a tensor for TensorBoard visualization
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def create_graph(self):
        tf.reset_default_graph()
        # 1st layer
        self.x_data_tf = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='x_data_tf')
        self.y_data_tf = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_data_tf')
        self.W_conv1_tf = self.weight_variable([self.s_f_conv1, self.s_f_conv1, 1, self.n_f_conv1], name='W_conv1_tf')
        self.b_conv1_tf = self.bias_variable([self.n_f_conv1], name = 'b_conv1_tf')
        self.h_conv1_tf = tf.nn.relu(self.conv2d(self.x_data_tf, self.W_conv1_tf) + self.b_conv1_tf, name = 'h_conv1_tf')
        self.h_pool1_tf = self.max_pool_2x2(self.h_conv1_tf, name = 'h_pool1_tf')
        # 2nd layer
        self.W_conv2_tf = self.weight_variable([self.s_f_conv2, self.s_f_conv2, self.n_f_conv1, self.n_f_conv2], name = 'W_conv2_tf')
        self.b_conv2_tf = self.bias_variable([self.n_f_conv2], name = 'b_conv2_tf')
        self.h_conv2_tf = tf.nn.relu(self.conv2d(self.h_pool1_tf, self.W_conv2_tf) + self.b_conv2_tf, name ='h_conv2_tf')
        self.h_pool2_tf = self.max_pool_2x2(self.h_conv2_tf, name = 'h_pool2_tf')
        # 3rd layer
        self.W_conv3_tf = self.weight_variable([self.s_f_conv3, self.s_f_conv3, self.n_f_conv2, self.n_f_conv3], name = 'W_conv3_tf')
        self.b_conv3_tf = self.bias_variable([self.n_f_conv3], name = 'b_conv3_tf')
        self.h_conv3_tf = tf.nn.relu(self.conv2d(self.h_pool2_tf, self.W_conv3_tf) + self.b_conv3_tf, name = 'h_conv3_tf') 
        self.h_pool3_tf = self.max_pool_2x2(self.h_conv3_tf, name = 'h_pool3_tf')
        # 4th layer
        self.W_fc1_tf = self.weight_variable([4 * 4 * self.n_f_conv3, self.n_n_fc1], name = 'W_fc1_tf')
        self.b_fc1_tf = self.bias_variable([self.n_n_fc1], name = 'b_fc1_tf')
        self.h_pool3_flat_tf = tf.reshape(self.h_pool3_tf, [-1, 4 * 4 * self.n_f_conv3], name = 'h_pool3_flat_tf')
        self.h_fc1_tf = tf.nn.relu(tf.matmul(self.h_pool3_flat_tf, self.W_fc1_tf) + self.b_fc1_tf, name = 'h_fc1_tf')
        # dropout
        self.keep_prob_tf = tf.placeholder(dtype=tf.float32, name = 'keep_prob_tf')
        self.h_fc1_drop_tf = tf.nn.dropout(self.h_fc1_tf, self.keep_prob_tf, name = 'h_fc1_drop_tf')
        # Output layer
        self.W_fc2_tf = self.weight_variable([self.n_n_fc1, 10], name = 'W_fc2_tf')
        self.b_fc2_tf = self.bias_variable([10], name = 'b_fc2_tf')
        self.z_pred_tf = tf.add(tf.matmul(self.h_fc1_drop_tf, self.W_fc2_tf), self.b_fc2_tf, name = 'z_pred_tf')
        # Cost function
        self.cross_entropy_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_data_tf, logits=self.z_pred_tf), name = 'cross_entropy_tf')
        # optimisation function
        self.learn_rate_tf = tf.placeholder(dtype=tf.float32, name="learn_rate_tf")
        self.train_step_tf = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(self.cross_entropy_tf, name = 'train_step_tf')
        # One hot encode
        self.y_pred_proba_tf = tf.nn.softmax(self.z_pred_tf, name='y_pred_proba_tf')
        self.y_pred_correct_tf = tf.equal(tf.argmax(self.y_pred_proba_tf, 1), tf.argmax(self.y_data_tf, 1), name = 'y_pred_correct_tf') 
        self.accuracy_tf = tf.reduce_mean(tf.cast(self.y_pred_correct_tf, dtype=tf.float32), name = 'accuracy_tf')
        # tensors to save immediate loss / accuracy during training
        self.train_loss_tf = tf.Variable(np.array([]), dtype=tf.float32, name='train_loss_tf', validate_shape = False)
        self.valid_loss_tf = tf.Variable(np.array([]), dtype=tf.float32, name='valid_loss_tf', validate_shape = False)
        self.train_acc_tf = tf.Variable(np.array([]), dtype=tf.float32, name='train_acc_tf', validate_shape = False)
        self.valid_acc_tf = tf.Variable(np.array([]), dtype=tf.float32, name='valid_acc_tf', validate_shape = False)
        return None

    def attach_summary(self, sess):
        # create summary tensors for tensorboard
        self.use_tb_summary = True
        self.summary_variable(self.W_conv1_tf, 'W_conv1_tf')
        self.summary_variable(self.b_conv1_tf, 'b_conv1_tf')
        self.summary_variable(self.W_conv2_tf, 'W_conv2_tf')
        self.summary_variable(self.b_conv2_tf, 'b_conv2_tf')
        self.summary_variable(self.W_conv3_tf, 'W_conv3_tf')
        self.summary_variable(self.b_conv3_tf, 'b_conv3_tf')
        self.summary_variable(self.W_fc1_tf, 'W_fc1_tf')
        self.summary_variable(self.b_fc1_tf, 'b_fc1_tf')
        self.summary_variable(self.W_fc2_tf, 'W_fc2_tf')
        self.summary_variable(self.b_fc2_tf, 'b_fc2_tf')
        tf.summary.scalar('cross_entropy_tf', self.cross_entropy_tf)
        tf.summary.scalar('accuracy_tf', self.accuracy_tf)
        # merge all summaries for tensorboard
        self.merged = tf.summary.merge_all()
        # initialize summary writer 
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        filepath = os.path.join(os.getcwd(), 'logs', (self.nn_name+'_'+timestamp))
        self.train_writer = tf.summary.FileWriter(os.path.join(filepath,'train'), sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(filepath,'valid'), sess.graph)    

    def attach_saver(self):
        # initialize tensorflow saver
        self.use_tf_saver = True
        self.saver_tf = tf.train.Saver()

    def train_graph(self, sess, x_train, y_train, x_valid, y_valid, n_epoch = 1, train_on_augmented_data = False):
        # train on original or augmented data
        self.train_on_augmented_data = train_on_augmented_data
        # training and validation data
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        # use augmented data
        if self.train_on_augmented_data:
            print('generate new set of images')
            self.x_train_aug = normalize_data(self.generate_images(self.x_train))
            self.y_train_aug = self.y_train
        # parameters
        mb_per_epoch = self.x_train.shape[0]/self.mb_size
        train_loss, train_acc, valid_loss, valid_acc = [],[],[],[]
        # start timer
        start = datetime.datetime.now()
        print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'),': start training')
        print('learnrate = ',self.learn_rate,', n_epoch = ', n_epoch, ', mb_size = ', self.mb_size)
        # looping over mini batches
        for i in range(int(n_epoch*mb_per_epoch) + 1):
            # adapt learn_rate
            self.learn_rate_pos = int(self.current_epoch // self.learn_rate_step_size)
            if not self.learn_rate == self.learn_rate_array[self.learn_rate_pos]:
                self.learn_rate = self.learn_rate_array[self.learn_rate_pos]
                print(datetime.datetime.now()-start,': set learn rate to %.6f'%self.learn_rate)
            # get new batch
            x_batch, y_batch = self.next_mini_batch() 
            # run the graph
            sess.run(self.train_step_tf, feed_dict={self.x_data_tf: x_batch, 
                                                    self.y_data_tf: y_batch, 
                                                    self.keep_prob_tf: self.keep_prob, 
                                                    self.learn_rate_tf: self.learn_rate})
            # store losses and accuracies
            if i % int(self.log_step * mb_per_epoch) == 0 or i == int(n_epoch * mb_per_epoch):
                self.n_log_step += 1 # for logging the results
                feed_dict_train = {
                    self.x_data_tf: self.x_train[self.perm_array[:len(self.x_valid)]], 
                    self.y_data_tf: self.y_train[self.perm_array[:len(self.y_valid)]], 
                    self.keep_prob_tf: 1.0}
                feed_dict_valid = {self.x_data_tf: self.x_valid, 
                                   self.y_data_tf: self.y_valid, 
                                   self.keep_prob_tf: 1.0}
                # summary for tensorboard
                if self.use_tb_summary:
                    train_summary = sess.run(self.merged, feed_dict = feed_dict_train)
                    valid_summary = sess.run(self.merged, feed_dict = feed_dict_valid)
                    self.train_writer.add_summary(train_summary, self.n_log_step)
                    self.valid_writer.add_summary(valid_summary, self.n_log_step)
                train_loss.append(sess.run(self.cross_entropy_tf, feed_dict = feed_dict_train))
                train_acc.append(self.accuracy_tf.eval(session = sess, feed_dict = feed_dict_train))
                valid_loss.append(sess.run(self.cross_entropy_tf, feed_dict = feed_dict_valid))
                valid_acc.append(self.accuracy_tf.eval(session = sess, feed_dict = feed_dict_valid))
                print('%.2f epoch: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(self.current_epoch, train_loss[-1],
                valid_loss[-1], train_acc[-1], valid_acc[-1]))
        # concatenate losses and accuracies and assign to tensor variables
        tl_c = np.concatenate([self.train_loss_tf.eval(session=sess), train_loss], axis = 0)
        vl_c = np.concatenate([self.valid_loss_tf.eval(session=sess), valid_loss], axis = 0)
        ta_c = np.concatenate([self.train_acc_tf.eval(session=sess), train_acc], axis = 0)
        va_c = np.concatenate([self.valid_acc_tf.eval(session=sess), valid_acc], axis = 0)
        sess.run(tf.assign(self.train_loss_tf, tl_c, validate_shape = False))
        sess.run(tf.assign(self.valid_loss_tf, vl_c , validate_shape = False))
        sess.run(tf.assign(self.train_acc_tf, ta_c , validate_shape = False))
        sess.run(tf.assign(self.valid_acc_tf, va_c , validate_shape = False))
        print('running time for training: ', datetime.datetime.now() - start)
        return None
    
    def save_model(self, sess):
        # tf saver
        if self.use_tf_saver:
            filepath = os.path.join(os.getcwd(), self.nn_name)
            self.saver_tf.save(sess, filepath)
        # tb summary
        if self.use_tb_summary:
            self.train_writer.close()
            self.valid_writer.close()
        return None

    def forward(self, sess, x_data):
        # forward prediction of current graph
        y_pred_proba = self.y_pred_proba_tf.eval(session = sess, feed_dict = {self.x_data_tf: x_data, self.keep_prob_tf: 1.0})
        return y_pred_proba
    
    def load_tensors(self, graph):
        # function to load tensors from a saved graph
        # input tensors
        self.x_data_tf = graph.get_tensor_by_name("x_data_tf:0")
        self.y_data_tf = graph.get_tensor_by_name("y_data_tf:0")
        # weights and bias tensors
        self.W_conv1_tf = graph.get_tensor_by_name("W_conv1_tf:0")
        self.W_conv2_tf = graph.get_tensor_by_name("W_conv2_tf:0")
        self.W_conv3_tf = graph.get_tensor_by_name("W_conv3_tf:0")
        self.W_fc1_tf = graph.get_tensor_by_name("W_fc1_tf:0")
        self.W_fc2_tf = graph.get_tensor_by_name("W_fc2_tf:0")
        self.b_conv1_tf = graph.get_tensor_by_name("b_conv1_tf:0")
        self.b_conv2_tf = graph.get_tensor_by_name("b_conv2_tf:0")
        self.b_conv3_tf = graph.get_tensor_by_name("b_conv3_tf:0")
        self.b_fc1_tf = graph.get_tensor_by_name("b_fc1_tf:0")
        self.b_fc2_tf = graph.get_tensor_by_name("b_fc2_tf:0")
        # activation tensors
        self.h_conv1_tf = graph.get_tensor_by_name('h_conv1_tf:0')  
        self.h_pool1_tf = graph.get_tensor_by_name('h_pool1_tf:0')
        self.h_conv2_tf = graph.get_tensor_by_name('h_conv2_tf:0')
        self.h_pool2_tf = graph.get_tensor_by_name('h_pool2_tf:0')
        self.h_conv3_tf = graph.get_tensor_by_name('h_conv3_tf:0')
        self.h_pool3_tf = graph.get_tensor_by_name('h_pool3_tf:0')
        self.h_fc1_tf = graph.get_tensor_by_name('h_fc1_tf:0')
        self.z_pred_tf = graph.get_tensor_by_name('z_pred_tf:0')
        # training and prediction tensors
        self.learn_rate_tf = graph.get_tensor_by_name("learn_rate_tf:0")
        self.keep_prob_tf = graph.get_tensor_by_name("keep_prob_tf:0")
        self.cross_entropy_tf = graph.get_tensor_by_name('cross_entropy_tf:0')
        self.train_step_tf = graph.get_operation_by_name('train_step_tf')
        self.z_pred_tf = graph.get_tensor_by_name('z_pred_tf:0')
        self.y_pred_proba_tf = graph.get_tensor_by_name("y_pred_proba_tf:0")
        self.y_pred_correct_tf = graph.get_tensor_by_name('y_pred_correct_tf:0')
        self.accuracy_tf = graph.get_tensor_by_name('accuracy_tf:0')
        # tensor of stored losses and accuricies during training
        self.train_loss_tf = graph.get_tensor_by_name("train_loss_tf:0")
        self.train_acc_tf = graph.get_tensor_by_name("train_acc_tf:0")
        self.valid_loss_tf = graph.get_tensor_by_name("valid_loss_tf:0")
        self.valid_acc_tf = graph.get_tensor_by_name("valid_acc_tf:0")
        return None
    
    def get_accuracy(self, sess):
        # get accuracies of training and validation sets
        train_acc = self.train_acc_tf.eval(session = sess)
        valid_acc = self.valid_acc_tf.eval(session = sess)
        return train_acc, valid_acc 
    
    def load_session_from_file(self, filename):
        # load session from file, restore graph, and load tensors
        tf.reset_default_graph()
        filepath = os.path.join(os.getcwd(), filename + '.meta')
        saver = tf.train.import_meta_graph(filepath)
        print(filepath)
        sess = tf.Session()
        saver.restore(sess, mn)
        graph = tf.get_default_graph()
        self.load_tensors(graph)
        return sess
    
def main():
    data_df = pd.read_csv("./data/train.csv")
    # extract and normalize images
    x_train_valid = data_df.iloc[:,1:].values.reshape(-1,28,28,1) # (42000,28,28,1) array
    x_train_valid = x_train_valid.astype(np.float) # convert from int64 to float32
    x_train_valid = normalize_data(x_train_valid)
    image_width = image_height = 28
    image_size = 784
    # extract image labels
    y_train_valid_labels = data_df.iloc[:,0].values # (42000,1) array
    labels_count = np.unique(y_train_valid_labels).shape[0]; # number of different labels = 10
    y_train_valid = dense_to_one_hot(y_train_valid_labels, labels_count).astype(np.uint8)

    # train the neural network graph
    nn_name = ['simpledigit']
    # cross validations
    cv_num = 10 # cross validations default = 20 => 5% validation set
    kfold = sklearn.model_selection.KFold(cv_num, shuffle=True, random_state=123)

    for i,(train_index, valid_index) in enumerate(kfold.split(x_train_valid)):
        # start timer
        start = datetime.datetime.now()
        # train and validation data of original images
        x_train = x_train_valid[train_index]
        y_train = y_train_valid[train_index]
        x_valid = x_train_valid[valid_index]
        y_valid = y_train_valid[valid_index]
        # create neural network graph
        nn_graph = simpleDigit(nn_name = nn_name[i])
        nn_graph.create_graph() # create graph
        nn_graph.attach_saver() # attach saver tensors
        # start tensorflow session
        with tf.Session() as sess:
            # attach summaries
            nn_graph.attach_summary(sess) 
            # variable initialization of the default graph
            sess.run(tf.global_variables_initializer()) 
            # training on original data
            nn_graph.train_graph(sess, x_train, y_train, x_valid, y_valid, n_epoch = 10.0)
            # training on augmented data
            #nn_graph.train_graph(sess, x_train, y_train, x_valid, y_valid, n_epoch = 14.0, train_on_augmented_data = True)
            # save tensors and summaries of model
            nn_graph.save_model(sess)
        # only one iteration
        if True:
            break
    print('total running time for training: ', datetime.datetime.now() - start)

main()
