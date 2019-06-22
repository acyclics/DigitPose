import cv2
import os
import pathlib

def convert():
    ''' Load images '''
    imagedir = os.path.join(".\\data", "MSRC_ObjCategImageDatabase_v1", "images")
    imagedir = pathlib.Path(imagedir)
    image_paths = list(imagedir.glob('*.*'))
    images = [cv2.imread(str(path)) for path in image_paths]

    ''' Write images '''
    print(str(image_paths[0])[0:-3] + "png")
    for idx, img in enumerate(images):
        cv2.imwrite(str(image_paths[idx])[0:-3] + "png", img)
    
    ''' Create labels '''
    labelsdir = os.path.join(".\\data", "MSRC_ObjCategImageDatabase_v1", "labels")
    labelsdir = pathlib.Path(labelsdir)
    label_paths = list(labelsdir.glob('*.*'))
    labels = [cv2.imread(str(path)) for path in label_paths]

    ''' Write labels '''
    for idx, label in enumerate(labels):
        cv2.imwrite(str(label_paths[idx])[0:-3] + "png", label)

convert()
