from dp_layers import DP
from dp_batch import Batch
import numpy as np
import os
import tensorflow as tf
import pathlib
from dp_houghVL import Hough
import cv2

IMAGE_HW = 224

def train_DP():
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
    poseCNN = DP(n_classes=n_classes, n_points=n_points, vgg16_npy_path=os.path.join(".\\data", "vgg16.npy"))
    poseCNN.build_graph()
    TRAIN_MODE = "labels"
    
    ''' Create Batch '''
    batch = Batch(image_paths, label_paths, label_paths2, label_paths3, n_classes=n_classes, n_points=n_points)
    N_epochs = 10000000
    N_eval = 10

    ''' Create Saver '''
    poseCNN.attach_saver()
    save_file = "./models/dp_train_weighted_justlabels_BW_11th/model.ckpt"
    LOAD_MODEL = True

    ''' Create TF session '''
    gpu_options = tf.GPUOptions(polling_inactive_delay_msecs=1000000, allow_growth=True)
    cfg = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=cfg) as sess:
        poseCNN.attach_summary(sess, "dp_train_weighted_justlabels_BW_11th", TRAIN_MODE)
        if not LOAD_MODEL:
            sess.run(tf.global_variables_initializer())
        else:
            poseCNN.saver_tf.restore(sess, save_file)
        avg_acc_label, avg_acc_center, avg_acc_pose = 0, 0, 0
        for epochs in range(N_epochs):
            RGB, LABEL, stack_centerxyz, oriens, coords = batch.get_image_and_label_ALL()
            if TRAIN_MODE == "labels":
                feed_dict = {poseCNN.RGB: RGB, poseCNN.LABEL: LABEL}
                sess.run(poseCNN.labels_train, feed_dict=feed_dict)
            elif TRAIN_MODE == "centers":
                feed_dict = {poseCNN.RGB: RGB, poseCNN.CENTER: stack_centerxyz}
                sess.run(poseCNN.centers_train, feed_dict=feed_dict)
            elif TRAIN_MODE == "pose":
                feed_dict = {poseCNN.RGB: RGB}
                labels_pred, directions = sess.run([poseCNN.labels_pred, poseCNN.c_conv8_1], feed_dict=feed_dict)
                directions = np.moveaxis(directions[0], -1, 0)
                hough_layer = Hough(n_classes, IMAGE_HW)
                hough_layer.cast_votes(labels_pred[0], directions[0], directions[1], directions[2])
                rois = hough_layer.get_rois()
                feed_dict = {poseCNN.RGB: RGB, poseCNN.ROIS: [rois], poseCNN.QUAT: oriens, poseCNN.COORDS: coords}
                sess.run(poseCNN.pose_train, feed_dict=feed_dict)
            else:
                print("Invalid TRAIN MODE")

            if (epochs + 1) % 100 == 0:
                if TRAIN_MODE == "labels" and not poseCNN.use_tb_summary:
                    avg_labels_acc = 0
                    for e in range(N_eval):
                        RGB, LABEL, stack_centerxyz, oriens, coords = batch.get_image_and_label_ALL()
                        feed_dict_labels = {poseCNN.RGB: RGB, poseCNN.LABEL: LABEL}
                        labels_acc = poseCNN.labels_pred_accuracy.eval(feed_dict=feed_dict_labels)
                        avg_labels_acc += labels_acc
                    print("Epoch:", epochs, "|  Labels accuracy =", avg_labels_acc / 10.0)
                elif TRAIN_MODE == "centers" and not poseCNN.use_tb_summary:
                    avg_centers_acc = 0
                    for e in range(N_eval):
                        RGB, LABEL, stack_centerxyz, oriens, coords = batch.get_image_and_label_ALL()
                        feed_dict_centers = {poseCNN.RGB: RGB, poseCNN.CENTER: stack_centerxyz}
                        centers_acc = poseCNN.centers_pred_accuracy.eval(feed_dict=feed_dict_centers)
                        avg_centers_acc += centers_acc
                    print("Epoch:", epochs, "|  Labels accuracy =", avg_centers_acc / 10.0)
                elif TRAIN_MODE == "pose" and not poseCNN.use_tb_summary:
                    feed_dict = {poseCNN.RGB: RGB}
                    labels_pred, directions = sess.run([poseCNN.labels_pred, poseCNN.c_conv8_1], feed_dict=feed_dict)
                    directions = np.moveaxis(directions[0], -1, 0)
                    hough_layer = Hough(n_classes, IMAGE_HW)
                    hough_layer.cast_votes(labels_pred[0], directions[0], directions[1], directions[2])
                    rois = hough_layer.get_rois()
                    feed_dict = {poseCNN.RGB: RGB, poseCNN.ROIS: [rois], poseCNN.QUAT: oriens, poseCNN.COORDS: coords}
                    pose_acc = sess.run(poseCNN.pose_pred_accuracy, feed_dict=feed_dict)
                    print("Epoch:", epochs, "|  Labels accuracy =", pose_acc)
                elif not poseCNN.use_tb_summary:
                    print("Invalid TRAIN MODE")

                if poseCNN.use_tb_summary:
                    if TRAIN_MODE == "labels":
                        feed_dict_train = {poseCNN.RGB: RGB, poseCNN.LABEL: LABEL}
                    elif TRAIN_MODE == "centers":
                        feed_dict_train = {poseCNN.RGB: RGB, poseCNN.CENTER: stack_centerxyz}
                    elif TRAIN_MODE == "pose":
                        feed_dict = {poseCNN.RGB: RGB}
                        labels_pred, directions = sess.run([poseCNN.labels_pred, poseCNN.c_conv8_1], feed_dict=feed_dict)
                        directions = np.moveaxis(directions[0], -1, 0)
                        hough_layer = Hough(n_classes, IMAGE_HW)
                        hough_layer.cast_votes(labels_pred[0], directions[0], directions[1], directions[2])
                        rois = hough_layer.get_rois()
                        feed_dict_train = {poseCNN.RGB: RGB, poseCNN.ROIS: [rois], poseCNN.QUAT: oriens, poseCNN.COORDS: coords}
                    train_summary = sess.run(poseCNN.merged, feed_dict=feed_dict_train)
                    poseCNN.train_writer.add_summary(train_summary, global_step=poseCNN.global_step.eval())

                poseCNN.saver_tf.save(sess, save_file)

train_DP()

'''
px, py, fx, fy = 0, 0, 2.77778, 2.77778
            coords = []
            for pt in centers:
                cx, cy, Tz = pt[0], pt[1], pt[2]
                Tx = ((cx - px) * Tz) / fx
                Ty = ((cy - py) * Tz) / fy
                coords.append([Tx, Ty, Tz])
            coords = [np.asarray(coords)]
'''
