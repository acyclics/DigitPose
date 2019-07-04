from dp_layers import DP
from dp_batch import Batch
import numpy as np
import os
import tensorflow as tf
import pathlib
from dp_houghVL import Hough
import cv2
from scipy.spatial.transform import Rotation as R

IMAGE_HW = 224

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
    poseCNN = DP(n_classes=n_classes, n_points=n_points, vgg16_npy_path=os.path.join(".\\data", "vgg16.npy"))
    poseCNN.build_graph()
    
    ''' Create Batch '''
    batch = Batch(image_paths, label_paths, label_paths2, label_paths3, n_classes=n_classes, n_points=n_points)
    N_epochs = 100000

    ''' Create Saver '''
    saver = tf.train.Saver(max_to_keep=1)

    ''' Create TF session '''
    gpu_options = tf.GPUOptions(polling_inactive_delay_msecs=1000000, allow_growth=True)
    cfg = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=cfg) as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./models/dp_train_weighted_justlabels_BW_11th/model.ckpt")
        avg_acc_label, avg_acc_center, avg_acc_pose = 0, 0, 0
        for epochs in range(N_epochs):
            RGB, LABEL, stack_centerxyz, oriens, coords = batch.get_image_and_label_ALL()
            feed_dict = {poseCNN.RGB: RGB, poseCNN.LABEL: LABEL}
            labels_pred = sess.run(poseCNN.labels_pred, feed_dict=feed_dict)
            #bgr = cv2.cvtColor(RGB[0].astype('float32'), cv2.COLOR_RGB2BGR)
            for x in range(224):
                for y in range(224):
                    if labels_pred[0][x][y] == 0:
                        cv2.circle(RGB[0], (x, y), 1, (0,0,255), -1)
            '''
            directions = np.moveaxis(directions[0], -1, 0)
            hough_layer = Hough(n_classes, 224)
            hough_layer.cast_votes(labels_pred[0], directions[0], directions[1], directions[2])
            centers = hough_layer.tally_votes()[0]
            rois = hough_layer.get_rois()[0]
            cv2.rectangle(bgr, (int(rois[0] * 224), int(rois[1] * 224)), (int(rois[2] * 224), int(rois[3] * 224)), (128, 0, 128))
            cv2.circle(bgr, (centers[0], centers[1]), 3, (0,255,0), -1)
            '''
            '''
            feed_dict_pose = {poseCNN.RGB: RGB, poseCNN.ROIS: [[rois]]}
            pose = sess.run(poseCNN.fc10_3, feed_dict=feed_dict_pose)
    
            points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
            r = R.from_quat(pose[0])
            r = np.matrix(r)
            r = cv2.UMat(r)
            cmtrix = [
                [2.77778, 0, 0],
                [0, 2.77778, 0],
                [0, 0, 1]
            ]
            axisPoints, _ = cv2.projectPoints(points, r, centers, cmtrix, (0, 0, 0, 0))
            bgr = cv2.line(bgr, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
            bgr = cv2.line(bgr, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
            bgr = cv2.line(bgr, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
            '''
            cv2.imshow("image", RGB[0])
            cv2.waitKey(0)

            '''
            RGB, LABEL, stack_centerxyz, oriens = batch.get_image_and_label_ALL()
            rois = [[0.0, 0.0, 0.9955357142857143, 0.9955357142857143]]
            coords = [np.asarray([[0., 1637.22296925, 20.48578928]])]
            poseinfo = [[114.9174, 118.569405, 720.6659, -1110.9646]]

            checker.test_loss_function(sess, poseinfo, oriens, coords)
            '''
            #print('Epoch {0}: Labels acc = {1:.4f}  |   Centers acc = {2:.4f}    |   Pose acc = {3:.4f}'.format(epochs, 
            #                                                                                             acc_labels,
            #                                                                                             acc_dir,
            #                                                                                             poseacc))






            '''
            feed_dict = {poseCNN.RGB: RGB, poseCNN.LABEL: LABEL}
            labels, directions = sess.run([poseCNN.labels_pred, poseCNN.c_conv8_1], feed_dict=feed_dict)
            bgr = cv2.cvtColor(RGB[0].astype('float32'), cv2.COLOR_RGB2BGR)
            #for x in range(224):
            #    for y in range(224):
            #        if LABEL[0][x][y][0] != 0:
            #            cv2.circle(bgr, (x, y), 1, (0,255,0), -1)
            for x in range(224):
                for y in range(224):
                    if labels[0][x][y] != 1:
                        cv2.circle(bgr, (x, y), 1, (0,0,255), -1)
            stack_centerxyz = np.moveaxis(stack_centerxyz[0], -1, 0)
            LABEL = poseCNN.argmax_label.eval(feed_dict, sess)
            hough_layer = Hough(n_classes, 224)
            hough_layer.cast_votes(LABEL[0], stack_centerxyz[0], stack_centerxyz[1], stack_centerxyz[2])
            centers = hough_layer.tally_votes()[0]
            rois = hough_layer.get_rois()[0]
            cv2.rectangle(bgr, (int(rois[0] * 224), int(rois[1] * 224)), (int(rois[2] * 224), int(rois[3] * 224)), (128, 0, 128))
            cv2.circle(bgr, (centers[0], centers[1]), 3, (0,255,0), -1)
            cv2.imshow("image", bgr)
            cv2.waitKey(0)
            '''

assess_DP()
