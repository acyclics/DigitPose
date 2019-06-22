'''
Papers: 
https://arxiv.org/pdf/1711.00199.pdf
'''
import tensorflow as tf
import numpy as np
import cv2

VGG_MEAN = [103.939, 116.779, 123.68]
IMAGE_HW = 224

''' General functions '''
''' END of General functions '''

class DP:
    def __init__(self, n_classes, vgg16_npy_path=None):        
        ''' Hyper-parameters '''
        self.labels_lr = 0.0001
        
        ''' VGG16 '''
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()

        ''' Object pose estimation '''
        self.n_classes = n_classes

        ''' Variables '''
        self.RGB = tf.placeholder(dtype=tf.float32, shape=[1, IMAGE_HW, IMAGE_HW, 3])
        self.LABEL = tf.placeholder(dtype=tf.float32, shape=[1, IMAGE_HW, IMAGE_HW, self.n_classes])
    
    def build_graph(self):
        ''' Create variables needed '''
        self.global_step = tf.train.get_or_create_global_step()

        ''' Build VGG16 '''
        rgb_scaled = self.rgb_to_255(self.RGB)
        bgr = self.rgb_to_bgr(rgb_scaled)

        self.conv1_1 = self.vgg16_conv_layer(bgr, "conv1_1")
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

        ''' VGG16 ends here; Labelling begins here '''
        self.filt6_1_1 = self.filter_variable([1, 1, 512, 64], "filt6_1_1")
        self.bias6_1_1 = self.bias_variable([64], "bias6_1_1")
        self.conv6_1_1 = self.conv_layer(self.conv5_3, self.filt6_1_1, self.bias6_1_1, "SAME", "conv6_1_1")

        # shape6_2 = [14, int(IMAGE_HW / 8), int(IMAGE_HW / 8), 64]
        self.filt6_2 = self.filter_variable([2, 2, 64, 64], "filt6_2")
        self.bias6_2 = self.bias_variable([64], "bias6_2")
        self.deconv6_2 = self.deconv_layer(self.conv6_1_1, self.filt6_2, 2, 64, "VALID", "deconv6_2")

        # shape6_1_2 = [28, int(IMAGE_HW / 8), int(IMAGE_HW / 8), 64]
        self.filt6_1_2 = self.filter_variable([2, 2, 64, 512], "filt6_1_2") # Note that for deconv, order is output channels then input channels
        self.bias6_1_2 = self.bias_variable([64], "bias6_1_2")
        self.deconv6_1_2 = self.deconv_layer(self.conv5_3, self.filt6_1_2, 2, 64, "VALID", "deconv6_1_2")

        self.add7_1 = tf.keras.layers.Add()([self.deconv6_2, self.deconv6_1_2])
        # shape7_2 = [28, IMAGE_HW, IMAGE_HW, 64]
        self.filt7_2 = self.filter_variable([8, 8, 64, 64], "filt7_2")
        self.bias7_2 = self.bias_variable([64], "bias7_2")
        self.deconv7_2 = self.deconv_layer(self.add7_1, self.filt7_2, 8, 64, "VALID", "deconv7_2")

        self.filt8_1 = self.filter_variable([1, 1, 64, self.n_classes], "filt8_1")
        self.bias8_1 = self.bias_variable([self.n_classes], "bias8_1")
        self.conv8_1 = self.conv_layer(self.deconv7_2, self.filt8_1, self.bias8_1, "SAME", "conv8_1")

        self.labels_probs = tf.nn.softmax(self.conv8_1, name="labels_probs", axis=-1)
        self.labels_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.LABEL, logits=self.conv8_1, axis=-1, name="labels_loss")
        #self.labels_loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(LABEL * tf.log(self.labels_probs + 1e-10), tf.ones(self.n_classes)), axis=[3]))
        self.labels_train = tf.train.AdamOptimizer(self.labels_lr).minimize(self.labels_loss, name = 'labels_train')
        #self.labels_train = tf.train.MomentumOptimizer(learning_rate=self.labels_lr, momentum=0.9, use_nesterov=True).minimize(self.labels_loss, global_step=self.global_step)
        self.labels_pred_correct = tf.equal(tf.argmax(self.labels_probs, -1), tf.argmax(self.LABEL, -1), name = 'labels_pred_correct') 
        self.labels_pred_accuracy = tf.reduce_mean(tf.cast(self.labels_pred_correct, dtype=tf.float32), name = 'labels_pred_accuracy')

        # Debug
        self.argmax = tf.argmax(self.labels_probs, -1)

        self.argmax_label = tf.argmax(self.LABEL, -1)
        self.argmax_label_shape = tf.shape(tf.argmax(self.LABEL, -1))

        self.argmax_shape = tf.shape(tf.argmax(self.labels_probs, -1))
        self.labels_probs_shape = tf.shape(self.labels_probs)

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
    
    def vgg16_get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")
    
    def vgg16_get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def vgg16_get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def filter_variable(self, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

