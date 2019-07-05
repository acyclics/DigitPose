from dp_layers import DP
from dp_batch import Batch
from dp_houghVL import Hough
import numpy as np
import os
import tensorflow as tf
import pathlib
import cv2

def assess_DP():
    ''' Load images '''
    imagedir = os.path.join(".\\data", "DigitPose-Frame", "Snapshots")
    imagedir = pathlib.Path(imagedir)
    image_paths = list(imagedir.glob('*.*'))
    image_paths = [str(path) for path in image_paths]

    ''' Create labels '''
    labelsdir = os.path.join(".\\data", "DigitPose-Frame", "Groundtruths")
    labelsdir = pathlib.Path(labelsdir)
    label_paths = list(labelsdir.glob('*'))
    label_paths = [str(path) for path in label_paths]

    ''' Create labels2 '''
    labelsdir2 = os.path.join(".\\data", "DigitPose-Frame", "Orientations")
    labelsdir2 = pathlib.Path(labelsdir2)
    label_paths2 = list(labelsdir2.glob('*'))
    label_paths2 = [str(path) for path in label_paths2]

    ''' Create labels3 '''
    labelsdir3 = os.path.join(".\\data", "DigitPose-Frame", "Points")
    labelsdir3 = pathlib.Path(labelsdir3)
    label_paths3 = list(labelsdir3.glob('*'))
    label_paths3 = [str(path) for path in label_paths3]

    ''' Create CNN '''
    n_classes = 2
    n_points = 1
    debug = False
    IMAGE_HW = 224
    model_dir = "./data/imagenet-vgg-verydeep-19.mat"
    poseCNN = DP(debug=debug, n_classes=n_classes, n_points=n_points, IMAGE_WH=IMAGE_HW, model_dir=model_dir)
    TRAIN_MODE = "labels"
    
    ''' Create Batch '''
    batch = Batch(image_paths, label_paths, label_paths2, label_paths3, n_classes=n_classes, n_points=n_points)
    N_epochs = 10000000
    N_eval = 10

    ''' Create Saver '''
    poseCNN.attach_saver(TRAIN_MODE)
    save_file = "./models/dp_train_new_1st/model.ckpt"
    LOAD_MODEL = True

    ''' Create TF session '''
    gpu_options = tf.GPUOptions(polling_inactive_delay_msecs=1000000, allow_growth=True)
    cfg = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=cfg) as sess:
        if not LOAD_MODEL:
            sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            poseCNN.saver_tf.restore(sess, save_file)
        for epochs in range(N_epochs):
            RGB, LABEL, stack_centerxyz, oriens, coords = batch.get_image_and_label_ALL()
            feed_dict_labels = {poseCNN.image: RGB, poseCNN.labels_keep_probability: 1.0}
            labels_pred = sess.run(poseCNN.labels_pred, feed_dict=feed_dict_labels)
            #bgr = cv2.cvtColor(RGB[0].astype('float32'), cv2.COLOR_RGB2BGR)
            for x in range(224):
                for y in range(224):
                    if labels_pred[0][x][y] == 1:
                        cv2.circle(RGB[0], (x, y), 1, (0,0,255), -1)
            cv2.imshow("image", RGB[0])
            cv2.waitKey(0)


assess_DP()
