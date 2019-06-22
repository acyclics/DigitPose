import random
import tensorflow as tf
import numpy as np
import cv2

IMAGE_HW = 224
colors = [ [0, 0, 0], [0, 128, 0], [0, 0, 128], [0, 128, 128], [128, 0, 0],
           [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64],
           [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192] ]

class Batch:
    def __init__(self, im_paths, im_labels):
        self.im_paths = im_paths
        self.im_labels = im_labels

    def get_image_and_label(self):
        sample = random.randint(0, len(self.im_paths) - 1)
        img, label = self.load_and_preprocess_image(self.im_paths[sample], self.im_labels[sample])
        return [img], [label]

    def preprocess_image(self, image):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_HW, IMAGE_HW))
        image = image / 255.0
        image = np.asarray(image)
        return image

    def preprocess_label(self, label_path):
        label = cv2.imread(label_path)
        stack = []
        for c in colors:
            mask = cv2.inRange(label, np.array(c), np.array(c)) / 255
            mask = cv2.resize(mask, (IMAGE_HW, IMAGE_HW))
            stack.append(mask)
        stack = np.asarray(stack)
        stack = np.moveaxis(stack, 0, -1)
        return stack

    def load_and_preprocess_image(self, path, label):
        return self.preprocess_image(path), self.preprocess_label(label)
