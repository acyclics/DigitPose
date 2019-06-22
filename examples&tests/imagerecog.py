import tensorflow as tf
import numpy as np
import random
import pathlib
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

''' Variables - training related '''
IMAGE_WIDTH, IMAGE_HEIGHT = 50, 50
''' Variables - training related '''

''' Functions - training related '''
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image /= 255.0
    return image

def load_and_preprocess_image(path, label):
    image = tf.read_file(path)
    return preprocess_image(image), label

def dense_to_one_hot(labels_dense, num_classes):
    # convert class labels from scalars to one-hot vectors e.g. 1 => [0 1 0 0 0 0 0 0 0 0]
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
''' END of Functions - training related '''

class SimpleDigit:
    def __init__(self, im_paths, im_labels):
        # Hyper-parameters
        self.mb_size = 50
        self.n_epochs = 3000
        self.s_f_conv1 = 3; # filter size of first convolution layer (default = 3)
        self.n_f_conv1 = 36; # number of features of first convolution layer (default = 36)
        self.s_f_conv2 = 3; # filter size of second convolution layer (default = 3)
        self.n_f_conv2 = 36; # number of features of second convolution layer (default = 36)
        self.s_f_conv3 = 3; # filter size of third convolution layer (default = 3)
        self.n_f_conv3 = 36; # number of features of third convolution layer (default = 36)
        self.n_n_fc1 = 576; # number of neurons of first fully connected layer (default = 576)

        self.learn_rate_tf = 0.0001
        self.keep_prob = 0.33 # keeping probability with dropout regularization 
        
        # Batch
        self.path_ds = tf.data.Dataset.from_tensor_slices((im_paths, im_labels))
        self.image_label_ds = self.path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds = self.image_label_ds.shuffle(buffer_size=len(im_paths))
        self.ds = self.ds.batch(self.mb_size)
        self.ds = self.ds.repeat(self.n_epochs)
        self.ds = self.ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.iterator = self.ds.make_one_shot_iterator()

    def weight_variable(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)
    
    def bias_variable(self, shape, name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W, name = None):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)

    def max_pool_2x2(self, x, name = None):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = name)

    def build_nn(self):
        x, y = self.iterator.get_next()
        # Layer 1
        self.W_conv1_tf = self.weight_variable([self.s_f_conv1, self.s_f_conv1, 3, self.n_f_conv1], name="W_conv1_tf")
        self.b_conv1_tf = self.bias_variable([self.n_f_conv1], name="b_conv1_tf")
        self.h_conv1_tf = tf.nn.relu(self.conv2d(x, self.W_conv1_tf) + self.b_conv1_tf, name="h_conv1_tf")
        self.h_pool1_tf = self.max_pool_2x2(self.h_conv1_tf, name="h_pool1_tf")
        # Layer 2
        self.W_conv2_tf = self.weight_variable([self.s_f_conv2, self.s_f_conv2, self.n_f_conv1, self.n_f_conv2], name = 'W_conv2_tf')
        self.b_conv2_tf = self.bias_variable([self.n_f_conv2], name = 'b_conv2_tf')
        self.h_conv2_tf = tf.nn.relu(self.conv2d(self.h_pool1_tf, self.W_conv2_tf) + self.b_conv2_tf, name ='h_conv2_tf')
        self.h_pool2_tf = self.max_pool_2x2(self.h_conv2_tf, name = 'h_pool2_tf')
        # Layer 3
        self.W_conv3_tf = self.weight_variable([self.s_f_conv3, self.s_f_conv3, self.n_f_conv2, self.n_f_conv3], name = 'W_conv3_tf')
        self.b_conv3_tf = self.bias_variable([self.n_f_conv3], name = 'b_conv3_tf')
        self.h_conv3_tf = tf.nn.relu(self.conv2d(self.h_pool2_tf, self.W_conv3_tf) + self.b_conv3_tf, name = 'h_conv3_tf') 
        self.h_pool3_tf = self.max_pool_2x2(self.h_conv3_tf, name = 'h_pool3_tf')
        # Layer 4
        filter_l4_span = 7 * 7 * self.n_f_conv3
        self.W_fc1_tf = self.weight_variable([filter_l4_span, self.n_n_fc1], name = 'W_fc1_tf')
        self.b_fc1_tf = self.bias_variable([self.n_n_fc1], name = 'b_fc1_tf')
        self.h_pool3_flat_tf = tf.reshape(self.h_pool3_tf, [-1, filter_l4_span], name = 'h_pool3_flat_tf')
        self.h_fc1_tf = tf.nn.relu(tf.matmul(self.h_pool3_flat_tf, self.W_fc1_tf) + self.b_fc1_tf, name = 'h_fc1_tf')
        # dropout
        self.keep_prob_tf = tf.placeholder(dtype=tf.float32, name = 'keep_prob_tf')
        self.h_fc1_drop_tf = tf.nn.dropout(self.h_fc1_tf, rate=1-self.keep_prob_tf, name = 'h_fc1_drop_tf')
        # Output layer
        self.W_fc2_tf = self.weight_variable([self.n_n_fc1, 5], name = 'W_fc2_tf')
        self.b_fc2_tf = self.bias_variable([5], name = 'b_fc2_tf')
        self.z_pred_tf = tf.add(tf.matmul(self.h_fc1_drop_tf, self.W_fc2_tf), self.b_fc2_tf, name = 'z_pred_tf')
        # Cost function
        self.cross_entropy_tf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=self.z_pred_tf), name = 'cross_entropy_tf')
        # optimisation function
        self.train_step_tf = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(self.cross_entropy_tf, name = 'train_step_tf')
        # One hot encode
        self.y_pred_proba_tf = tf.nn.softmax(self.z_pred_tf, name='y_pred_proba_tf')
        self.y_pred_correct_tf = tf.equal(tf.argmax(self.y_pred_proba_tf, 1), tf.argmax(y, 1), name = 'y_pred_correct_tf') 
        self.accuracy_tf = tf.reduce_mean(tf.cast(self.y_pred_correct_tf, dtype=tf.float32), name = 'accuracy_tf')
        # tensors to save immediate loss / accuracy during training
        self.train_loss_tf = tf.Variable(np.array([]), dtype=tf.float32, name='train_loss_tf', validate_shape = False)
        self.valid_loss_tf = tf.Variable(np.array([]), dtype=tf.float32, name='valid_loss_tf', validate_shape = False)
        self.train_acc_tf = tf.Variable(np.array([]), dtype=tf.float32, name='train_acc_tf', validate_shape = False)
        self.valid_acc_tf = tf.Variable(np.array([]), dtype=tf.float32, name='valid_acc_tf', validate_shape = False)
        return None
    
    def train(self, sess):
        train_loss, train_acc = [], []
        for e in range(self.n_epochs):
            try:
                sess.run(self.train_step_tf, feed_dict={self.keep_prob_tf: self.keep_prob})
                
                feed_dict_train = {self.keep_prob_tf: 1.0}
                train_loss.append(sess.run(self.cross_entropy_tf, feed_dict = feed_dict_train))
                train_acc.append(self.accuracy_tf.eval(session = sess, feed_dict = feed_dict_train))
                print('%.2f epoch: train loss = %.4f, train acc = %.4f'%(e, train_loss[-1], train_acc[-1]))
                
            except tf.errors.OutofRangeError:
                break
            
def main():
    # Loading images
    flowerdir = os.path.join(os.getcwd(), 'data', 'flower_photos')
    flowerdir = pathlib.Path(flowerdir)
    all_image_paths = list(flowerdir.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    # Create labels
    label_names = sorted(item.name for item in flowerdir.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = np.array([label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths])
    labels_count = len(label_names)
    y_train = dense_to_one_hot(all_image_labels, labels_count).astype(np.uint8)
    sd = SimpleDigit(all_image_paths, y_train)
    sd.build_nn()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        sd.train(sess)
    
main()
