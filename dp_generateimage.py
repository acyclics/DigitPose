import keras.preprocessing.image
import cv2

def generate_images(imgs):
    # rotations, translations, zoom
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20, width_shift_range=0.5 , height_shift_range=0.5, brightness_range=(0.5, 1.0)
        zoom_range = 0.5)
    # get transformed images
    imgs = image_generator.flow(imgs.copy(), np.zeros(len(imgs)),
                                batch_size=len(imgs), shuffle = False).next()    
    return imgs[0]
