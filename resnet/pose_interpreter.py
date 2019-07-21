from resnet_model import Model
from resnet_utils import get_pc
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tfquaternion as tfq

class PointsL1Loss():

    def __init__(self, numpy_pcs, n_classes, batch_size, m_points, use_negative_qr_loss=False):
        self.numpy_pcs = np.array(numpy_pcs)
        self.use_negative_qr_loss = use_negative_qr_loss
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.m_points = m_points
    
    def to_transform(self, q, pos):
        q = tf.squeeze(q, axis=-1)
        pos = tf.squeeze(pos, axis=-1)
        rot = tfq.Quaternion(q).as_rotation_matrix()
        r1 = tf.concat([rot[:, 0], pos[:, 0:1]], axis=-1)
        r2 = tf.concat([rot[:, 1], pos[:, 1:2]], axis=-1)
        r3 = tf.concat([rot[:, 2], pos[:, 2:3]], axis=-1)
        r4 = [[0, 0, 0, 1] for _ in range(self.batch_size)]
        trans = tf.stack([r1, r2, r3, r4], axis=-1)
        return trans
    
    def true_transform(self, q, pos):
        rot = tfq.Quaternion(q).as_rotation_matrix()
        r1 = tf.concat([rot[:, 0], pos[:, 0:1]], axis=-1)
        r2 = tf.concat([rot[:, 1], pos[:, 1:2]], axis=-1)
        r3 = tf.concat([rot[:, 2], pos[:, 2:3]], axis=-1)
        r4 = [[0, 0, 0, 1] for _ in range(self.batch_size)]
        trans = tf.stack([r1, r2, r3, r4], axis=-1)
        return trans

    def compute_loss(self, x, transt, transp):
        x = [x]
        x = tf.transpose(x, [1, 0])
        diff = tf.math.subtract(tf.matmul(transt[0], x), tf.matmul(transp[0], x))
        return tf.math.abs(diff)

    def compute(self, qtrue, qpred, postrue, pospred, object_index):
        four = tf.constant(4, shape=[1])
        three = tf.constant(3, shape=[1])
        zero = tf.constant(0.0, dtype=tf.float32, shape=[1])
        transt = self.true_transform(qtrue, postrue)
        qp = tf.gather(qpred, [tf.math.multiply(four, object_index),
                               tf.math.multiply(four, object_index) + 1,
                               tf.math.multiply(four, object_index) + 2,
                               tf.math.multiply(four, object_index) + 3], axis=1)
        pp = tf.gather(pospred, [tf.math.multiply(three, object_index),
                               tf.math.multiply(three, object_index) + 1,
                               tf.math.multiply(three, object_index) + 2], axis=1)
        transp = self.to_transform(qp, pp)

        new_numpy_pcs = []
        self.numpy_pcs = np.transpose(self.numpy_pcs, [0, 2, 1])
        for i in range(len(self.numpy_pcs)):
            indices = []
            for j in range(self.m_points):
                randind = np.random.randint(0, len(self.numpy_pcs[i]))
                indices.append(self.numpy_pcs[i][randind])
            new_numpy_pcs.append(indices)
        new_numpy_pcs = np.transpose(np.asarray(new_numpy_pcs), [0, 2, 1])

        pcs = tf.convert_to_tensor(new_numpy_pcs, tf.float32)
        pcs = tf.squeeze(tf.squeeze(tf.gather(pcs, [object_index], axis=0), axis=0), axis=0)
        pcs = tf.transpose(pcs, [1, 0])
        all_losses = tf.map_fn(lambda x: self.compute_loss(x, transt, transp), pcs)
  
        bp_loss = tf.reduce_mean(all_losses)
        #if self.use_negative_qr_loss:
        #    bp_loss = tf.add(bp_loss, tf.maximum(-qpred[:, 0:1], zero))
        return bp_loss

class PoseInterpreter:

    def __init__(self, n_classes, batch_size, m_points, pcd_dir):
        ''' Variables '''
        self.n_classes = n_classes
        numpy_pcs = [get_pc(pcd_dir, str(n + 1) + ".pcd") for n in range(n_classes)]

        self.PointsL1Loss = PointsL1Loss(numpy_pcs=numpy_pcs, n_classes=n_classes,
                                         batch_size=batch_size, m_points=m_points, use_negative_qr_loss=True)

        self.mask = tf.placeholder(tf.float32, [None, 224, 224, 1], name="mask")
        self.object_index = tf.placeholder(tf.int32, [1], name="object_index")
        self.qt = tf.placeholder(tf.float32, shape=[None, 4], name="qt")
        self.pt = tf.placeholder(tf.float32, shape=[None, 3], name="pt")
        
        ''' Hyper-parameters '''
        self.lr = 1e-4

        ''' Build NN '''
        self.build_graph()
        self.build_train()

    def build_graph(self):
        model = Model(resnet_size=18, bottleneck=False, num_classes=5, num_filters=32,
               kernel_size=7,
               conv_stride=2, first_pool_size=3, first_pool_stride=2,
               block_sizes=[64, 128, 256, 512], block_strides=[1, 2, 2, 2], data_format=None)

        resnet_dim = 256 * 7 * 7

        resnet18 = tf.reshape(model(self.mask, True), [-1, resnet_dim])
        self.multilayer_percep = self.fc_layer_with_relu(resnet18, [resnet_dim, 256], [256], scope="shared", name="multilayer_percep")
    
    def build_position_branch(self):
        with tf.variable_scope("position"):
            fc_1 = self.fc_layer(self.multilayer_percep, [256, 3 * self.n_classes], [3 * self.n_classes], scope="position", name="fc_1")
        return fc_1

    def build_orientation_branch(self):
        with tf.variable_scope("orientation"):
            fc_1 = self.fc_layer(self.multilayer_percep, [256, 4 * self.n_classes], [4 * self.n_classes], scope="orientation", name="fc_1")
            for i in range(self.n_classes):
                q_normalzied = tf.div(fc_1[:, i*4:(i+1)*4], tf.norm(fc_1[:, i*4:(i+1)*4]))
                if i == 0:
                    fc_1_normalzied = q_normalzied
                else:
                    fc_1_normalzied = tf.concat([fc_1_normalzied, q_normalzied], axis=-1)
        return fc_1_normalzied

    def build_train(self):
        self.position_pred = self.build_position_branch()
        self.orientation_pred = self.build_orientation_branch()
        loss = self.PointsL1Loss.compute(qtrue=self.qt, qpred=self.orientation_pred, postrue=self.pt, pospred=self.position_pred, object_index=self.object_index)
        self.dloss = loss
        four = tf.constant(4, shape=[1])
        three = tf.constant(3, shape=[1])
        orien_pred = tf.gather(self.orientation_pred, [tf.math.multiply(four, self.object_index),
                               tf.math.multiply(four, self.object_index) + 1,
                               tf.math.multiply(four, self.object_index) + 2,
                               tf.math.multiply(four, self.object_index) + 3], axis=1)
        pos_pred = tf.gather(self.position_pred, [tf.math.multiply(three, self.object_index),
                               tf.math.multiply(three, self.object_index) + 1,
                               tf.math.multiply(three, self.object_index) + 2], axis=1)
        self.pos_acc = tf.reduce_mean(tf.cast(tf.equal(pos_pred, self.pt), tf.float32))
        self.orien_acc = tf.reduce_mean(tf.cast(tf.equal(orien_pred, self.qt), tf.float32))
        self.train = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def fc_layer_with_relu(self, x, w_shape, b_shape, scope, name):
        with tf.variable_scope(scope):
            initializer = tf.contrib.layers.xavier_initializer()
            w = tf.Variable(initializer(w_shape))
            b = tf.Variable(initializer(b_shape))
            relu = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w), b), name=name)
        return relu
    
    def fc_layer(self, x, w_shape, b_shape, scope, name):
        with tf.variable_scope(scope):
            initializer = tf.contrib.layers.xavier_initializer()
            w = tf.Variable(initializer(w_shape))
            b = tf.Variable(initializer(b_shape))
            layer = tf.nn.bias_add(tf.matmul(x, w), b)
        return layer
