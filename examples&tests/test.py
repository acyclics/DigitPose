import tensorflow as tf
import random
import pathlib
import os
import numpy as np
import cv2

def preprocess_label(label, colors):
    label = cv2.imread(label)
    stack = []
    for c in colors:
        mask = cv2.inRange(label, np.array(c), np.array(c)) / 255
        mask = cv2.resize(mask, (224, 224))
        print(mask)
        stack.append(mask)
    stack = np.asarray(stack)
    tensor = tf.reshape(tf.image.convert_image_dtype(stack, tf.uint8), [224, 224, len(colors)])
    return tf.concat(tensor, axis=-1)

''' Create labels '''
labelsdir = os.path.join("..\\data", "MSRC_ObjCategImageDatabase_v1", "labels")
labelsdir = pathlib.Path(labelsdir)
label_paths = list(labelsdir.glob('*.*'))
label_paths = [str(path) for path in label_paths]

colors = [ [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128] ]

tensor = preprocess_label(label_paths[0], colors)
