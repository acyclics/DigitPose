from dp_layers import DP
from dp_batch import Batch
import numpy as np
import os
import tensorflow as tf
import pathlib

IMAGE_HW = 224

def train_DP():
    ''' Load images '''
    imagedir = os.path.join(".\\data", "MSRC_ObjCategImageDatabase_v1", "images_png")
    imagedir = pathlib.Path(imagedir)
    image_paths = list(imagedir.glob('*.*'))
    image_paths = [str(path) for path in image_paths]

    ''' Create labels '''
    labelsdir = os.path.join(".\\data", "MSRC_ObjCategImageDatabase_v1", "labels_png")
    labelsdir = pathlib.Path(labelsdir)
    label_paths = list(labelsdir.glob('*.*'))
    label_paths = [str(path) for path in label_paths]

    ''' Create CNN '''
    n_classes = 14
    poseCNN = DP(n_classes=n_classes, vgg16_npy_path=os.path.join(".\\data", "vgg16.npy"))
    poseCNN.build_graph()

    ''' Create Batch '''
    batch = Batch(image_paths, label_paths)
    N_epochs = 100000

    ''' Create Saver '''
    saver = tf.train.Saver(max_to_keep=5)

    ''' Create TF session '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_acc = 0
        for epochs in range(N_epochs):
            RGB, LABEL = batch.get_image_and_label()
            feed_dict = {poseCNN.RGB: RGB, poseCNN.LABEL: LABEL}
            sess.run(poseCNN.labels_train, feed_dict=feed_dict)
            acc = poseCNN.labels_pred_accuracy.eval(session = sess, feed_dict=feed_dict)
            avg_acc += acc
            if (epochs + 1) % 100 == 0:
                saver.save(sess, './models/dp_train_2nd/model', global_step=poseCNN.global_step)
                print('epoch: train acc = %.4f'%(avg_acc / 100.0))
                avg_acc = 0

train_DP()
