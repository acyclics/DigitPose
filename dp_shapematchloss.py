from scipy.spatial.transform import Rotation as R
import tensorflow as tf
import numpy as np
import tfquaternion as tfq

def tf_truncate(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def SLoss(q_true, q_pred, M, n_classes, no_of_points):
    losses = []
    M = tf.transpose(M, [0, 2, 1])
    for i in range(0, 4 * n_classes, 4):
        GTR = tfq.Quaternion(tf.slice(q_true, [0, i], [1, 4])).as_rotation_matrix()
        ESTR = tfq.Quaternion(tf.slice(q_pred, [0, i], [1, 4])).as_rotation_matrix()
        ALL_ESTR = tf.matmul(ESTR[0], M[0])
        ALL_GTR = tf.matmul(GTR[0], M[0])
        for k in range(1, no_of_points):
            ALL_ESTR = tf.concat([ALL_ESTR, tf.matmul(ESTR[0], M[k])], axis=0)
            ALL_GTR = tf.concat([ALL_GTR, tf.matmul(GTR[0], M[k])], axis=0)
        totloss = []
        if no_of_points > 1:
            for j in range(no_of_points):
                subloss = tf.multiply(tf.norm(tf.subtract(ALL_ESTR[j], ALL_GTR[0])), tf.norm(tf.subtract(ALL_ESTR[j], ALL_GTR[0])))
                for n in range(no_of_points):
                    subloss = tf.minimum(tf.multiply(tf.norm(tf.subtract(ALL_ESTR[j], ALL_GTR[n])), tf.norm(tf.subtract(ALL_ESTR[j], ALL_GTR[n]))), subloss)
                totloss.append(subloss)
        else:
            subloss = tf.multiply(tf.norm(tf.subtract(ALL_ESTR, ALL_GTR)), tf.norm(tf.subtract(ALL_ESTR, ALL_GTR)))
            totloss.append(subloss)
        retloss = totloss[0]
        for l in range(1, n_classes):
            retloss = tf.concat([retloss, totloss[l]], axis=0)
        loss = tf.divide(tf.reduce_sum(retloss), 2 * no_of_points)
        losses.append(loss)
    finalLoss = losses[0]
    for f in range(1, n_classes):
        finalLoss = tf.concat([finalLoss, losses[f]], axis=0)
    return tf.reduce_mean(finalLoss, name="pose_loss")

def SLoss_accuracy(q_true, q_pred, n_classes):
    losses = []
    for i in range(0, 4 * n_classes, 4):
        GTR = tfq.Quaternion(tf.slice(q_true, [0, i], [1, 4])).as_rotation_matrix()
        ESTR = tfq.Quaternion(tf.slice(q_pred, [0, i], [1, 4])).as_rotation_matrix()
        equals = tf.equal(tf_truncate(GTR, 1), tf_truncate(ESTR, 1))
        losses.append(tf.reduce_mean(tf.cast(equals, dtype=tf.float32)))
    acc = losses[0]
    for l in range(1, len(losses)):
        acc = tf.add(acc, losses[l])
    return acc

class SML:
    def __init__(self):
        pass

    def SLoss(self, q_true, q_pred, M, n_classes, no_of_points):
        losses = []
        M = tf.transpose(M, [0, 2, 1])
        for i in range(0, 4 * n_classes, 4):
            GTR = tfq.Quaternion(tf.slice(q_true, [0, i], [1, 4])).as_rotation_matrix()
            ESTR = tfq.Quaternion(tf.slice(q_pred, [0, i], [1, 4])).as_rotation_matrix()
            ALL_ESTR = tf.matmul(ESTR[0], M[0])
            ALL_GTR = tf.matmul(GTR[0], M[0])
            for k in range(1, no_of_points):
                ALL_ESTR = tf.concat([ALL_ESTR, tf.matmul(ESTR[0], M[k])], axis=0)
                ALL_GTR = tf.concat([ALL_GTR, tf.matmul(GTR[0], M[k])], axis=0)
            totloss = []
            if no_of_points > 1:
                for j in range(no_of_points):
                    subloss = tf.multiply(tf.norm(tf.subtract(ALL_ESTR[j], ALL_GTR[0])), tf.norm(tf.subtract(ALL_ESTR[j], ALL_GTR[0])))
                    for n in range(no_of_points):
                        subloss = tf.minimum(tf.multiply(tf.norm(tf.subtract(ALL_ESTR[j], ALL_GTR[n])), tf.norm(tf.subtract(ALL_ESTR[j], ALL_GTR[n]))), subloss)
                    totloss.append(subloss)
            else:
                subloss = tf.multiply(tf.norm(tf.subtract(ALL_ESTR, ALL_GTR)), tf.norm(tf.subtract(ALL_ESTR, ALL_GTR)))
                totloss.append(subloss)
            retloss = totloss[0]
            for l in range(1, n_classes):
                retloss = tf.concat([retloss, totloss[l]], axis=0)
            loss = tf.divide(tf.reduce_sum(retloss), 2 * no_of_points)
            losses.append(loss)
        finalLoss = losses[0]
        for f in range(1, n_classes):
            finalLoss = tf.concat([finalLoss, losses[f]], axis=0)
        return tf.reduce_mean(finalLoss)

    def test_loss_function(self):
        n_classes = 2
        ''' Pose branch variables '''
        QUAT_EST = tf.placeholder(dtype=tf.float32, shape=[1, 4 * (n_classes - 1)])
        QUAT_TRU = tf.placeholder(dtype=tf.float32, shape=[1, 4 * (n_classes - 1)])
        COORDS = tf.placeholder(dtype=tf.float32, shape=[n_classes - 1, 1, 3])
        loss = self.SLoss(QUAT_TRU, QUAT_EST, COORDS, n_classes - 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            QE = [[1, 1, 1, 1]]
            QT = [[0, 0, 0, 1]]
            C = [[[5, 5, 10]]]
            feed_dict = {QUAT_EST: QE, QUAT_TRU: QT, COORDS: C}
            while True:
                a = sess.run(self.LOSS, feed_dict=feed_dict)
                print(a)

    def test_loss_function1(self):
        n_classes = 2
        ''' Pose branch variables '''
        QUAT_EST = tf.placeholder(dtype=tf.float32, shape=[1, 4 * (n_classes - 1)])
        QUAT_TRU = tf.placeholder(dtype=tf.float32, shape=[1, 4 * (n_classes - 1)])
        COORDS = tf.placeholder(dtype=tf.float32, shape=[n_classes - 1, 1, 3])
        loss = self.SLoss(QUAT_TRU, QUAT_EST, COORDS, n_classes - 1, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            QE = [[1, 1, 1, 1]]
            QT = [[0, 0, 0, 1]]
            C = [[[5, 5, 10]]]
            feed_dict = {QUAT_EST: QE, QUAT_TRU: QT, COORDS: C}
            while True:
                a= sess.run([self.LOSS], feed_dict=feed_dict)
                print(a)
    
    def test_loss_function(self, sess, QE, QT, C):
        n_classes = 2
        ''' Pose branch variables '''
        QUAT_EST = tf.placeholder(dtype=tf.float32, shape=[1, 4 * (n_classes - 1)])
        QUAT_TRU = tf.placeholder(dtype=tf.float32, shape=[1, 4 * (n_classes - 1)])
        COORDS = tf.placeholder(dtype=tf.float32, shape=[n_classes - 1, 1, 3])
        loss = self.SLoss(QUAT_TRU, QUAT_EST, COORDS, n_classes - 1)
        feed_dict = {QUAT_EST: QE, QUAT_TRU: QT, COORDS: C}
        a = sess.run(self.LOSS, feed_dict=feed_dict)
        print(a)
