'''
Papers: 
https://arxiv.org/pdf/1711.00199.pdf
'''
import tensorflow as tf
import numpy as np
import cv2
from dp_roipool import ROIPoolingLayer
from dp_shapematchloss import SLoss, SLoss_accuracy
from dp_labelsloss import weighted_PixelWise_CrossEntropy
import sys
import datetime
import os

VGG_MEAN = [103.939, 116.779, 123.68]

''' General functions '''
''' END of General functions '''

class DP:
    def __init__(self, n_classes, n_points, IMAGE_HW=224, vgg16_npy_path=None):        
        ''' Hyper-parameters '''
        self.labels_lr = 0.001
        self.labels_mm = 0.9
        self.labels_dropout = 0.2
        self.labels_gClip = 5.0
        self.labels_l2_alpha = 0.01
        self.labels_posw = 100000.0
        self.centers_lr = 0.00001
        self.centers_mm = 0.9
        self.centers_gClip = 5.0
        self.pose_lr = 0.00001
        self.pose_mm = 0.9
        self.pose_gClip = 5.0
        self.roi_pool_h = 7
        self.roi_pool_w = 7
        self.no_of_points = n_points
        self.TRUNCATE = 0
        
        ''' VGG16 '''
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()

        ''' Tensorflow utils '''
        self.use_tb_summary = False
        self.use_tf_saver = False

        ''' Object pose estimation '''
        self.n_classes = n_classes
        self.IMAGE_HW = IMAGE_HW

        ''' Labels branch variables '''
        self.RGB = tf.placeholder(dtype=tf.float32, shape=[1, IMAGE_HW, IMAGE_HW, 3])
        self.LABEL = tf.placeholder(dtype=tf.float32, shape=[1, IMAGE_HW, IMAGE_HW, self.n_classes])

        ''' Centers branch variables '''
        self.CENTER = tf.placeholder(dtype=tf.float32, shape=[1, IMAGE_HW, IMAGE_HW, 3 * (self.n_classes - 1)])

        ''' Pose branch variables '''
        self.ROIS = tf.placeholder(dtype=tf.float32, shape=[1, None, 4])
        self.QUAT = tf.placeholder(dtype=tf.float32, shape=[1, 4 * (self.n_classes - 1)])
        self.COORDS = tf.placeholder(dtype=tf.float32, shape=[n_points, 1, 3])
    
    def build_graph(self):
        ''' Create variables needed '''
        self.global_step = tf.train.get_or_create_global_step()

        ''' Build VGG16 '''
        #rgb_scaled = self.rgb_to_255(self.RGB)
        #bgr = self.rgb_to_bgr(rgb_scaled)

        self.conv1_1 = self.vgg16_conv_layer(self.RGB, "conv1_1")
        self.conv1_2 = self.vgg16_conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.vgg16_max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.vgg16_conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.vgg16_conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.vgg16_max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.vgg16_conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.vgg16_conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.vgg16_conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.vgg16_max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.vgg16_conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.vgg16_conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.vgg16_conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.vgg16_max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.vgg16_conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.vgg16_conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.vgg16_conv_layer(self.conv5_2, "conv5_3")

        ''' VGG16 ends here; Branches begins here '''
        self.build_labels_branch()
        #self.build_centers_branch()
        #self.build_pose_branch()
        #self.build_observation_nodes()

    def build_labels_branch(self):
        self.filt6_1_1 = self.filter_variable([1, 1, 512, 64], "filt6_1_1")
        self.bias6_1_1 = self.bias_variable([64], "bias6_1_1")
        self.conv6_1_1 = self.conv_layer(self.conv5_3, self.filt6_1_1, self.bias6_1_1, "SAME", "conv6_1_1")
        #self.conv6_1_1 = tf.nn.dropout(self.conv6_1_1, self.labels_dropout)

        self.filt6_2 = self.filter_variable([2, 2, 64, 64], "filt6_2")
        self.bias6_2 = self.bias_variable([64], "bias6_2")
        self.deconv6_2 = self.deconv_layer(self.conv6_1_1, self.filt6_2, 2, 64, "SAME", "deconv6_2")

        self.filt6_1_2 = self.filter_variable([1, 1, 512, 64], "filt6_1_2") # Note that for deconv, order is output channels then input channels
        self.bias6_1_2 = self.bias_variable([64], "bias6_1_2")
        self.conv6_1_2 = self.conv_layer(self.conv4_3, self.filt6_1_2, self.bias6_1_2, "SAME", "conv6_1_2")
        #self.conv6_1_2 = tf.nn.dropout(self.conv6_1_2, self.labels_dropout)

        self.add7_1 = tf.keras.layers.Add()([self.deconv6_2, self.conv6_1_2])
        self.filt7_2 = self.filter_variable([8, 8, 64, 64], "filt7_2")
        self.bias7_2 = self.bias_variable([64], "bias7_2")
        self.deconv7_2 = self.deconv_layer(self.add7_1, self.filt7_2, 8, 64, "SAME", "deconv7_2")

        self.filt8_1 = self.filter_variable([1, 1, 64, self.n_classes], "filt8_1")
        self.bias8_1 = self.bias_variable([self.n_classes], "bias8_1")
        self.conv8_1 = self.conv_layer(self.deconv7_2, self.filt8_1, self.bias8_1, "SAME", "conv8_1")

        #self.labels_sigmoid = tf.math.sigmoid(self.conv8_1, name="labels_sigmoid")
        self.labels_probs = tf.nn.softmax(self.conv8_1, name="labels_probs")
        self.labels_loss = weighted_PixelWise_CrossEntropy(labels=self.LABEL, logits=self.labels_probs) #+ self.labels_l2_alpha * (tf.nn.l2_loss(self.filt8_1) + tf.nn.l2_loss(self.filt7_2) + tf.nn.l2_loss(self.filt6_1_2) + tf.nn.l2_loss(self.filt6_2) + tf.nn.l2_loss(self.filt6_1_1))
        #self.labels_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.LABEL[:, :, :, 0:-1], logits=self.labels_sigmoid, pos_weight=self.labels_posw, name="labels_loss"))
        #self.labels_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.LABEL, logits=self.conv8_1, name="labels_loss")) + self.labels_l2_alpha * (tf.nn.l2_loss(self.filt8_1) + tf.nn.l2_loss(self.filt7_2) + tf.nn.l2_loss(self.filt6_1_2) + tf.nn.l2_loss(self.filt6_2) + tf.nn.l2_loss(self.filt6_1_1))
        optimizer = tf.train.AdamOptimizer(self.labels_lr) #.minimize(self.labels_loss, name = 'labels_train')
        #optimizer = tf.train.MomentumOptimizer(learning_rate=self.labels_lr, momentum=self.labels_mm, use_nesterov=True)
        gradients, variables = zip(*optimizer.compute_gradients(self.labels_loss, var_list=[
            self.bias8_1, self.filt8_1, self.bias7_2, self.filt7_2, self.bias6_1_2, self.filt6_1_2,
            self.bias6_2, self.filt6_2, self.bias6_1_1, self.filt6_1_1
        ]))
        gradients, _ = tf.clip_by_global_norm(gradients, self.labels_gClip)
        self.labels_train = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
        self.labels_pred_correct = tf.equal(tf.argmax(self.labels_probs, -1), tf.argmax(self.LABEL, -1), name = 'labels_pred_correct') 
        self.labels_pred_accuracy = tf.reduce_mean(tf.cast(self.labels_pred_correct, dtype=tf.float32), name = 'labels_pred_accuracy')
        #self.labels_pred_correct = tf.equal(self.conv8_1, self.LABEL[:, :, :, 0:-1], name = 'labels_pred_correct') 
        #self.labels_pred_accuracy = tf.reduce_mean(tf.cast(self.labels_pred_correct, dtype=tf.float32), name = 'labels_pred_accuracy')
        self.labels_pred = tf.argmax(self.labels_probs, -1)
        self.argmax_label = tf.argmax(self.LABEL, -1)

    def build_centers_branch(self):
        self.c_filt6_1_1 = self.filter_variable([1, 1, 512, 128], "c_filt6_1_1")
        self.c_bias6_1_1 = self.bias_variable([128], "c_bias6_1_1")
        self.c_conv6_1_1 = self.conv_layer(self.conv5_3, self.c_filt6_1_1, self.c_bias6_1_1, "SAME", "c_conv6_1_1")

        self.c_filt6_2 = self.filter_variable([2, 2, 128, 128], "c_filt6_2")
        self.c_bias6_2 = self.bias_variable([128], "c_bias6_2")
        self.c_deconv6_2 = self.deconv_layer(self.c_conv6_1_1, self.c_filt6_2, 2, 128, "SAME", "c_deconv6_2")

        self.c_filt6_1_2 = self.filter_variable([1, 1, 512, 128], "c_filt6_1_2") # Note that for deconv, order is output channels then input channels
        self.c_bias6_1_2 = self.bias_variable([128], "c_bias6_1_2")
        self.c_conv6_1_2 = self.conv_layer(self.conv4_3, self.c_filt6_1_2, self.c_bias6_1_2, "SAME", "c_conv6_1_2")

        self.c_add7_1 = tf.keras.layers.Add()([self.c_deconv6_2, self.c_conv6_1_2])
        self.c_filt7_2 = self.filter_variable([8, 8, 128, 128], "c_filt7_2")
        self.c_bias7_2 = self.bias_variable([128], "c_bias7_2")
        self.c_deconv7_2 = self.deconv_layer(self.c_add7_1, self.c_filt7_2, 8, 128, "SAME", "c_deconv7_2")

        self.c_filt8_1 = self.filter_variable([1, 1, 128, 3 * self.n_classes], "c_filt8_1")
        self.c_bias8_1 = self.bias_variable([3 * self.n_classes], "c_bias8_1")
        self.c_conv8_1 = self.conv_layer(self.c_deconv7_2, self.c_filt8_1, self.c_bias8_1, "SAME", "c_conv8_1")

        self.centers_loss = tf.losses.absolute_difference(labels=self.CENTER, predictions=self.c_conv8_1)
        #optimizer = tf.train.AdamOptimizer(self.centers_lr) #.minimize(self.centers_loss, name="centers_train")
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.centers_lr, momentum=self.centers_mm, use_nesterov=True) #.minimize(self.centers_loss, global_step=self.global_step)
        gradients, variables = zip(*optimizer.compute_gradients(self.centers_loss, var_list=[
            self.c_bias8_1, self.c_filt8_1, self.c_bias7_2, self.c_filt7_2, self.c_bias6_1_2, self.c_filt6_1_2, self.c_bias6_2,
            self.c_filt6_2, self.c_bias6_1_1, self.c_filt6_1_1
        ]))
        gradients, _ = tf.clip_by_global_norm(gradients, self.centers_gClip)
        self.centers_train = optimizer.apply_gradients(zip(gradients, variables))
        self.centers_pred_correct = tf.equal(self.c_conv8_1, self.CENTER, name = "centers_pred_correct") 
        self.centers_pred_accuracy = tf.reduce_mean(tf.cast(self.centers_pred_correct, dtype=tf.float32), name="centers_pred_accuracy")

    def build_pose_branch(self):
        self.roi_layer9_1 = ROIPoolingLayer(self.roi_pool_h, self.roi_pool_w)
        self.pooled_features9_1 = self.roi_layer9_1([self.conv5_3, self.ROIS])
        self.pooled_features9_1 = tf.where(tf.is_nan(self.pooled_features9_1), tf.ones_like(self.pooled_features9_1) * self.TRUNCATE, self.pooled_features9_1)
        self.pooled_features9_1 = tf.where(tf.is_inf(self.pooled_features9_1), tf.ones_like(self.pooled_features9_1) * self.TRUNCATE, self.pooled_features9_1)
        self.roi_layer9_2 = ROIPoolingLayer(self.roi_pool_h, self.roi_pool_w)
        self.pooled_features9_2 = self.roi_layer9_2([self.conv4_3, self.ROIS])
        self.pooled_features9_2 = tf.where(tf.is_nan(self.pooled_features9_2), tf.ones_like(self.pooled_features9_2) * self.TRUNCATE, self.pooled_features9_2)
        self.pooled_features9_2 = tf.where(tf.is_inf(self.pooled_features9_2), tf.ones_like(self.pooled_features9_2) * self.TRUNCATE, self.pooled_features9_2)
        self.roi_add9_3 = tf.keras.layers.Add()([self.pooled_features9_1, self.pooled_features9_2])
        self.fc10_1 = self.vgg16_fc_layer_forRoI(self.roi_add9_3, "fc6")
        self.fc10_2 = self.vgg16_fc_layer_forRoI_NONFIRST(self.fc10_1, "fc7")
        self.W_fc10_3 = self.filter_variable([4096, 4 * (self.n_classes - 1)], name="W_fc10_3")
        self.b_fc10_3 = self.bias_variable([4 * (self.n_classes - 1)], name="b_fc10_3")
        self.fc10_3 = self.fc_layer(self.fc10_2, self.W_fc10_3, self.b_fc10_3, "fc10_3")
        self.pose_loss = SLoss(self.QUAT, self.fc10_3, self.COORDS, self.n_classes - 1, self.no_of_points)
        #optimizer = tf.train.AdamOptimizer(self.pose_lr) #.minimize(self.pose_loss, name="pose_train")
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.pose_lr, momentum=self.pose_mm, use_nesterov=True)
        gradients, variables = zip(*optimizer.compute_gradients(self.pose_loss, var_list=[
            self.b_fc10_3, self.W_fc10_3
        ]))
        gradients, _ = tf.clip_by_global_norm(gradients, self.pose_gClip)
        self.pose_train = optimizer.apply_gradients(zip(gradients, variables))
        self.pose_pred_accuracy = SLoss_accuracy(self.QUAT, self.fc10_3, self.n_classes - 1)

    def build_observation_nodes(self):
        self.argmax_label = tf.argmax(self.LABEL, -1)

    def rgb_to_255(self, rgb):
        # Converts rgb from 0-1 to 0-255
        return rgb * 255.0
    
    def rgb_to_bgr(self, rgb):
        r, g, b = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        bgr = tf.concat(axis=3, values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]])
        return bgr

    def vgg16_avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def vgg16_max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def vgg16_conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.vgg16_get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.vgg16_get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def conv_layer(self, bottom, filter, bias, padding, name):
        conv = tf.nn.conv2d(bottom, filter, strides=[1, 1, 1, 1], padding=padding, name=name)
        bias = tf.nn.bias_add(conv, bias)
        relu = tf.nn.relu(bias)
        return relu

    def deconv_layer(self, bottom, filter, stride, out_channels, padding, name):
        in_shape = tf.shape(bottom, out_type=tf.dtypes.int32)
        h = in_shape[1] * stride
        w = in_shape[2] * stride
        new_shape = [in_shape[0], h, w, out_channels]
        deconv = tf.nn.conv2d_transpose(bottom, output_shape=new_shape, filter=filter, strides=[1, stride, stride, 1], padding=padding, name=name)
        return deconv

    def vgg16_fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            weights = self.vgg16_get_fc_weight(name)
            biases = self.vgg16_get_bias(name)
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc
    
    def vgg16_fc_layer_forRoI(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[2:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            weights = self.vgg16_get_fc_weight(name)
            biases = self.vgg16_get_bias(name)
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc
    
    def vgg16_fc_layer_forRoI_NONFIRST(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            weights = self.vgg16_get_fc_weight(name)
            biases = self.vgg16_get_bias(name)
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def fc_layer(self, x, weights, bias, name):
        with tf.variable_scope(name):
            fc = tf.nn.bias_add(tf.matmul(x, weights), bias)
        return fc

    def vgg16_get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")
    
    def vgg16_get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def vgg16_get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def filter_variable(self, shape, name=None):
        #initial = tf.truncated_normal(shape, stddev=0.1)
        #return tf.Variable(initial, name=name)
        initializer = tf.contrib.layers.xavier_initializer()
        W_xavier = tf.Variable(initializer(shape))
        return W_xavier

    def bias_variable(self, shape, name=None):
        #initial = tf.constant(0.1, shape=shape)
        #return tf.Variable(initial, name=name)
        initializer = tf.contrib.layers.xavier_initializer()
        B_xavier = tf.Variable(initializer(shape))
        return B_xavier
    
    def attach_summary(self, sess, dir_name, TRAIN_MODE):
        self.use_tb_summary = True
        if TRAIN_MODE == "labels":
            tf.summary.scalar('labels_loss', self.labels_loss)
            tf.summary.scalar('labels_accuracy', self.labels_pred_accuracy)
        elif TRAIN_MODE == "centers":
            tf.summary.scalar('centers_loss', self.centers_loss)
            tf.summary.scalar('centers_accuracy', self.centers_pred_accuracy)
        elif TRAIN_MODE == "pose":
            tf.summary.scalar('pose_loss', self.pose_loss)
            tf.summary.scalar('pose_accuracy', self.pose_pred_accuracy)
        else:
            print("Invalid TRAIN MODE")
        self.merged = tf.summary.merge_all()
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        filepath = os.path.join(os.getcwd(), 'logs', str(dir_name), timestamp)
        self.train_writer = tf.summary.FileWriter(filepath)
    
    def attach_saver(self):
        self.use_tf_saver = True
        self.saver_tf = tf.train.Saver(max_to_keep=2)
    