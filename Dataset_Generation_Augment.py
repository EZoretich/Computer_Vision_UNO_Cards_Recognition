# -------------------------------------------------- GENERATING NEW IMAGES FOR BIGGER DATASET
# In this code, thanks to the use of keras, a various set of new card images will be generated.
# New images will be useful to improve the accuracy of our Machinene Learning model

# ---------------- IMPORTING LIBRARIES

from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2  
import glob
import csv

# ---------------- CLASS DEFINITION

# Custom class to create object  containing image and image name (as cv2.imread() does not store filename)
class MyImage:
    def __init__(self, img_name):
        self.imgo = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

# -------------------------------------------------- MAIN CODE
# ---------------- DEFINE VARIABLES

kernel = np.ones((3,3), np.uint8)
filenames = glob.glob('C:/Users/elena/Desktop/UNO_Cards/*.jpg')
sep = '.'
images = [MyImage(img) for img in filenames]
j = 0
save_to_dir='C:/Users/elena/Desktop/UNO_Cards/'


for img in images:
    cards = img.imgo.reshape((1,) + img.imgo.shape)
    card_augment = ImageDataGenerator(rotation_range=30, brightness_range=(0.5, 1.5), shear_range=9.0) # selecting parameter for card augmentation
    card_augment.fit(cards)

    i = 0
    # keras build in function .flow() loads the images and generates batches of augmented data with specified names and formats.
    for batch in card_augment.flow(cards, batch_size=1, save_to_dir='C:/Users/elena/Desktop/UNO_Cards/', save_prefix="q"+str(img).split(sep, 1)[0].rsplit('\\', 1)[1], save_format='jpg'):
        j +=1
        i += 1
        if i > 5: # Create 6 variation of the same cards, then proceed to next one
            break
