from dp_layers import DP
from dp_batch import Batch
import numpy as np
import os
import tensorflow as tf
import pathlib
from dp_houghVL import Hough

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
    
    ''' Create Batch '''
    batch = Batch(image_paths, label_paths, label_paths2, label_paths3, n_classes=n_classes, n_points=n_points)
    N_epochs = 10000000
    N_eval = 10

    ''' Create Saver '''
    saver = tf.train.Saver(max_to_keep=1)
    
    ''' Info '''
    max_labels, max_centers, max_pose = (0.0, 0), (0.0, 0), (0.0, 0)

    ''' Create TF session '''
    gpu_options = tf.GPUOptions(polling_inactive_delay_msecs=1000000, allow_growth=True)
    cfg = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        avg_acc_label, avg_acc_center, avg_acc_pose = 0, 0, 0
        for epochs in range(N_epochs):
            RGB, LABEL, stack_centerxyz, oriens, coords = batch.get_image_and_label_ALL()
            feed_dict = {poseCNN.RGB: RGB, poseCNN.LABEL: LABEL, poseCNN.CENTER: stack_centerxyz}
            _, _, labels_pred, directions = sess.run([poseCNN.labels_train,
                                                      poseCNN.centers_train, poseCNN.labels_pred,
                                                      poseCNN.c_conv8_1], feed_dict=feed_dict)
            directions = np.moveaxis(directions[0], -1, 0)
            hough_layer = Hough(n_classes, IMAGE_HW)
            hough_layer.cast_votes(labels_pred[0], directions[0], directions[1], directions[2])
            rois = hough_layer.get_rois()
            feed_dict = {poseCNN.RGB: RGB, poseCNN.ROIS: [rois], poseCNN.QUAT: oriens, poseCNN.COORDS: coords}
            sess.run(poseCNN.pose_train, feed_dict)

            if (epochs + 1) % 100 == 0:
                RGB, LABEL, stack_centerxyz, oriens, coords = batch.get_image_and_label_ALL()
                feed_dict_labels = {poseCNN.RGB: RGB, poseCNN.LABEL: LABEL}
                labels_acc = poseCNN.labels_pred_accuracy.eval(feed_dict=feed_dict_labels)
                feed_dict_centers = {poseCNN.RGB: RGB, poseCNN.CENTER: stack_centerxyz}
                centers_acc = poseCNN.centers_pred_accuracy.eval(feed_dict=feed_dict_centers)
                feed_dict_pose = {poseCNN.RGB: RGB, poseCNN.ROIS: [rois], poseCNN.QUAT: oriens, poseCNN.COORDS: coords}
                pose_acc = poseCNN.pose_pred_accuracy.eval(feed_dict=feed_dict_pose)
                print("Epoch: {0}    |   Labels acc: {1:.4f}   |   Centers acc: {2:.4f}    |   Pose acc: {3:.4f}".format(epochs, labels_acc, centers_acc, pose_acc))
                saver.save(sess, './models/dp_train_1000imgs/model.ckpt', global_step=poseCNN.global_step)

                if max_labels[0] <= labels_acc:
                    max_labels = (labels_acc, epochs)
                if max_centers[0] <= centers_acc:
                    max_centers = (centers_acc, epochs)
                if max_pose[0] <= pose_acc:
                    max_pose = (pose_acc, epochs)
                
                print("BESTS|   Labels acc: {0:.4f}, epochs: {1}  |   Centers acc: {2:.4f}, epochs: {3}  |   Pose acc: {4:.4f}, epochs: {5}".format(max_labels[0], max_labels[1], max_centers[0], max_centers[1], max_pose[0], max_pose[1]))

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
