import sys
sys.path.insert(0, "../")
from pose_interpreter import PoseInterpreter
from dp_batch_1st import Batch
import numpy as np
import os
import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pathlib
import cv2

def train():
    ''' Load Snapshots '''
    imagedir = os.path.join("..\\data", "DigitPose-Frame", "Snapshots")
    imagedir = pathlib.Path(imagedir)
    image_paths = list(imagedir.glob('*.*'))
    image_paths = [str(path) for path in image_paths]

    ''' Load Groundtruths '''
    gtdir = os.path.join("..\\data", "DigitPose-Frame", "Groundtruths")
    gtdir = pathlib.Path(gtdir)
    gt_paths = list(gtdir.glob('*'))
    gt_paths = [str(path) for path in gt_paths]

    ''' Load Orientations '''
    oriendir = os.path.join("..\\data", "DigitPose-Frame", "Orientations")
    oriendir = pathlib.Path(oriendir)
    orien_paths = list(oriendir.glob('*'))
    orien_paths = [str(path) for path in orien_paths]

    ''' Load Points '''
    pts_dir = os.path.join("..\\data", "DigitPose-Frame", "Points")
    pts_dir = pathlib.Path(pts_dir)
    pts_paths = list(pts_dir.glob('*'))
    pts_paths = [str(path) for path in pts_paths]

    ''' Load Numbers '''
    nums_dir = os.path.join("..\\data", "DigitPose-Frame", "Numbers")
    nums_dir = pathlib.Path(nums_dir)
    num_paths = list(nums_dir.glob('*'))
    num_paths = [str(path) for path in num_paths]

    ''' Build models '''
    n_classes = 8
    batch_size = 1
    n_points = 1
    m_points = 1000
    pcd_dir = "./pcds"
    poseNN = PoseInterpreter(n_classes=n_classes, batch_size=batch_size, m_points=m_points, pcd_dir=pcd_dir)

    ''' Create Batch '''
    batch = Batch(im_paths=image_paths, im_labels=gt_paths, im_orientations=orien_paths, im_coordinates=pts_paths, im_numbers=num_paths,
                  n_classes=n_classes + 1, n_points=n_points)
    N_epochs = 10000000
    N_eval = 10

    ''' Create TF session '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for epochs in range(N_epochs):
            RGB, LABEL, stack_centerxyz, mask, oriens, coords, num = batch.get_image_and_label_ALL()
            oriens = [np.asarray([oriens[0][3], oriens[0][0], oriens[0][1], oriens[0][2]])]
            feed_dict = {poseNN.mask: mask, poseNN.qt: oriens, poseNN.pt: coords[0], poseNN.object_index: num}
            sess.run(poseNN.train, feed_dict=feed_dict)

            if (epochs + 1) % 10 == 0:
                RGB, LABEL, stack_centerxyz, mask, oriens, coords, num = batch.get_image_and_label_ALL()
                oriens = [np.asarray([oriens[0][3], oriens[0][0], oriens[0][1], oriens[0][2]])]
                feed_dict = {poseNN.mask: mask, poseNN.qt: oriens, poseNN.pt: coords[0], poseNN.object_index: num}
                loss, pos_acc, orien_acc = sess.run([poseNN.dloss, poseNN.pos_acc, poseNN.orien_acc], feed_dict=feed_dict)
                print("Epoch:   {0}     |       Position accuracy: {1:.4f}   |      Orientations accuracy: {2:.4f}".format(epochs + 1, pos_acc, orien_acc))
                print("Loss:", loss)

train()
