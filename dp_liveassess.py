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

def LiveAssess_DP():
    ''' Load images '''
    imagedir = os.path.join(".\\data", "real")
    imagedir = pathlib.Path(imagedir)
    image_paths = list(imagedir.glob('*'))
    image_paths = [str(path) for path in image_paths]

    ''' Create CNN '''
    n_classes = 2
    n_points = 1
    poseCNN = DP(n_classes=n_classes, n_points=n_points, vgg16_npy_path=os.path.join(".\\data", "vgg16.npy"))
    poseCNN.build_graph()
    
    ''' Create opencv camera '''
    #cap = cv2.VideoCapture(0)

    ''' Create Saver '''
    poseCNN.attach_saver()
    save_file = "./models/dp_train_weighted_justlabels_BW_11th/model.ckpt"

    ''' Create TF session '''
    gpu_options = tf.GPUOptions(polling_inactive_delay_msecs=1000000, allow_growth=True)
    cfg = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=cfg) as sess:
        poseCNN.saver_tf.restore(sess, save_file)
        for i in range(len(image_paths)):
            #ret, bgr = cap.read()
            bgr = cv2.imread(image_paths[i])
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            bgr = cv2.merge((bgr, bgr, bgr))
            RGB = cv2.resize(bgr, (IMAGE_HW, IMAGE_HW))
            #RGB = [cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IMAGE_HW, IMAGE_HW))]
            feed_dict = {poseCNN.RGB: [RGB]}
            labels_pred = sess.run(poseCNN.labels_pred, feed_dict=feed_dict)
            for x in range(224):
                for y in range(224):
                    if labels_pred[0][x][y] == 0:
                        cv2.circle(bgr, (x, y), 1, (0,0,255), -1)
            cv2.imshow('frame',bgr)
            cv2.waitKey(0)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

LiveAssess_DP()
