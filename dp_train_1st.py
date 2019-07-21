from dp_layers_1st import DP
from dp_batch_1st import Batch
import numpy as np
import os
import tensorflow as tf
import pathlib
import cv2

def train_DP():
    ''' Load Snapshots '''
    imagedir = os.path.join(".\\data", "DigitPose-Frame", "Snapshots")
    imagedir = pathlib.Path(imagedir)
    image_paths = list(imagedir.glob('*.*'))
    image_paths = [str(path) for path in image_paths]

    ''' Load Groundtruths '''
    gtdir = os.path.join(".\\data", "DigitPose-Frame", "Groundtruths")
    gtdir = pathlib.Path(gtdir)
    gt_paths = list(gtdir.glob('*'))
    gt_paths = [str(path) for path in gt_paths]

    ''' Load Orientations '''
    oriendir = os.path.join(".\\data", "DigitPose-Frame", "Orientations")
    oriendir = pathlib.Path(oriendir)
    orien_paths = list(oriendir.glob('*'))
    orien_paths = [str(path) for path in orien_paths]

    ''' Load Points '''
    pts_dir = os.path.join(".\\data", "DigitPose-Frame", "Points")
    pts_dir = pathlib.Path(pts_dir)
    pts_paths = list(pts_dir.glob('*'))
    pts_paths = [str(path) for path in pts_paths]

    ''' Load Numbers '''
    nums_dir = os.path.join(".\\data", "DigitPose-Frame", "Numbers")
    nums_dir = pathlib.Path(nums_dir)
    num_paths = list(nums_dir.glob('*'))
    num_paths = [str(path) for path in num_paths]

    ''' Create CNN '''
    n_classes = 8
    n_points = 1
    debug = False
    IMAGE_HW = 224
    model_dir = "./data/imagenet-vgg-verydeep-19.mat"
    poseCNN = DP(debug=debug, n_classes=n_classes, IMAGE_WH=IMAGE_HW, model_dir=model_dir)
    
    ''' Create Batch '''
    batch = Batch(im_paths=image_paths, im_labels=gt_paths, im_orientations=orien_paths, im_coordinates=pts_paths, im_numbers=num_paths,
                  n_classes=n_classes + 1, n_points=n_points)
    N_epochs = 10000000
    N_eval = 10

    ''' Create Saver '''
    save_file = "./models/dp_train_1st/model.ckpt"
    LOAD_MODEL = True
    poseCNN.attach_saver()
    
    ''' Create TF session '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if not LOAD_MODEL:
            sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            poseCNN.saver_tf.restore(sess, save_file)

        for epochs in range(N_epochs):
            RGB, LABEL, stack_centerxyz, mask, oriens, coords, num = batch.get_image_and_label_ALL()
            feed_dict = {poseCNN.image: RGB, poseCNN.labels_annotation: LABEL, poseCNN.labels_keep_probability: 0.85}
            sess.run(poseCNN.labels_train_op, feed_dict=feed_dict)

            if (epochs + 1) % 100 == 0:
                avg_labels_acc = 0
                RGB, LABEL, stack_centerxyz, mask, oriens, coords, num = batch.get_image_and_label_ALL()
                feed_dict_labels = {poseCNN.image: RGB, poseCNN.labels_annotation: LABEL, poseCNN.labels_keep_probability: 1.0}
                labels_acc = poseCNN.labels_pred_accuracy.eval(feed_dict=feed_dict_labels)
                avg_labels_acc += labels_acc
                print("Epoch:", epochs, "|  Labels accuracy =", avg_labels_acc / 1.0)
                poseCNN.saver_tf.save(sess, save_file)

train_DP()
