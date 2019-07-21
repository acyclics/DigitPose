from dp_layers_1st import DP
from dp_batch_1st import Batch
import numpy as np
import os
import tensorflow as tf
import pathlib
import cv2

def assess_DP():
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
    LOAD_MODEL = False
    poseCNN.attach_saver()
    
    ''' Create TF session '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.initializers.global_variables())
        poseCNN.saver_tf.restore(sess, save_file)

        for epochs in range(N_epochs):
            RGB, LABEL, stack_centerxyz, mask, oriens, coords, num = batch.get_image_and_label_ALL()
            feed_dict = {poseCNN.image: RGB, poseCNN.labels_keep_probability: 1.0}
            labels_pred = sess.run(poseCNN.labels_pred, feed_dict=feed_dict)
            bgr = cv2.cvtColor(RGB[0].astype('float32'), cv2.COLOR_RGB2BGR)
            for x in range(224):
                for y in range(224):
                    if labels_pred[0][x][y] == 1:
                        cv2.circle(RGB[0], (x, y), 1, (0,0,255), -1)
                    elif labels_pred[0][x][y] == 2:
                        cv2.circle(RGB[0], (x, y), 1, (255,0,255), -1)
                    elif labels_pred[0][x][y] == 3:
                        cv2.circle(RGB[0], (x, y), 1, (0,255,255), -1)
                    elif labels_pred[0][x][y] == 4:
                        cv2.circle(RGB[0], (x, y), 1, (125,125,125), -1)
                    elif labels_pred[0][x][y] == 5:
                        cv2.circle(RGB[0], (x, y), 1, (255,0,0), -1)
                    elif labels_pred[0][x][y] == 6:
                        cv2.circle(RGB[0], (x, y), 1, (0,255,0), -1)
                    elif labels_pred[0][x][y] == 7:
                        cv2.circle(RGB[0], (x, y), 1, (125,0,125), -1)
                    elif labels_pred[0][x][y] == 8:
                        cv2.circle(RGB[0], (x, y), 1, (125,125,125), -1)
            cv2.imshow("image", RGB[0])
            cv2.waitKey(0)

assess_DP()
