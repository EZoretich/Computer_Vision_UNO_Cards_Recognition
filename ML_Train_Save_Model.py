# TRAIN AND SAVE MACHINE LEARNING MODEL
# In this code, the 280 uno cards are going to be processed, the central number countours detected,
# and the appropriate features for Machine Learning are going to be stored.
# Those features will then be used to train the Random Forest Classifier, and the model will be
# saved in a joblib file.

# -------------------------------------------------------------- IMPORTING LIBRARIES
from itertools import repeat
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2  
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, subplots, show
import glob
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# -------------------------------------------------------------- DEFINING FUNCTIONS

# -------------------- DISPLAY IMAGES
def win_plot(image, im_title = 'bitmap_image'):
    cv2.imshow(im_title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(im_title)
    return()

# -------------------- CARD PROCESSING AND FEATURE EXTRACTION
def obt_features(card):
    card_bw = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    thr_value, card_th = cv2.threshold(card_bw, 185, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(card_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    sensitivity =  55
    hsv_min = np.array([0,0,255-sensitivity])
    hsv_max = np.array([255,sensitivity,255])

    card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    card_hsv_th = cv2.inRange(card_hsv, hsv_min, hsv_max)

    card_col_close = cv2.morphologyEx(card_hsv_th, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(card_col_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i,c in enumerate(contours):
        perimeter = cv2.arcLength(c, True)
        if perimeter >190 and perimeter < 420:
            area = cv2.contourArea(c)
            epsilon = 0.02*perimeter
            vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))
            ellipse = cv2.fitEllipse(c)
            (xc,yc),(d1,d2),angle = ellipse
            axes_ratio = round(d1/d2, 3)
            convexHull = cv2.convexHull(c)
            CH_len = len(convexHull)
            geometry = [perimeter, area, vertex_approx, axes_ratio, CH_len]
            [x,y,w,h] = cv2.boundingRect(c)
            cropped = card[y-15:y+h+15, x-15:x+w+15]
            cv2.drawContours(card, contours, -1, (0,0,0), 1)
            cv2.ellipse(card, ellipse, (0, 0, 0), 2)
    return geometry

# -------------------------------------------------------------- MAIN CODE

# --------------------  DEFINING  VARIABLES
kernel = np.ones((3,3), np.uint8)
ml_data = []
labels = []

# --------------------  ADDING CARD LABELS
# This will create a list of numbers corrisponding to the label of each uno card.
for l1 in range(4):
    for l2 in range(10):
        labels.append(l2)
for l3 in range(4):
    for l4 in range(0,10):
        labels.extend(repeat(l4, 6))

# --------------------  LOOPING THROUGH ALL IMAGES
# Passing the function for obtaining features though all cards,
# Then adding the array of selected features into another array (ml_data)
# Necessary for training the Machie Learning model
filenames = glob.glob('C:/Users/elena/Desktop/UNO_Cards/Dataset/*.jpg')
images = [cv2.imread(img) for img in filenames]
for img in images:
    ml_data.append(obt_features(img))

ml_data = np.array(ml_data)


X = ml_data
y = labels

# --------------------  DEFINING THE MACHINE LEARNING MODEL
# After the selection of the model, the features and labels arrays are going to be
# split into training and testing sets.
# ±86%  of cards will be used for training, and the remaining 14% will be use for testing

model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
# The train_test_split parameter 'shuffle = True' will allow to randomly select the cards (features)
# used for training and testing, providing a non-bias model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.14, shuffle=True) 
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
score = model.score(X_test, y_test)
print(score)
for h in range(len(y_predict)):
    print(y_predict[h], y_test[h])

# --------------------  SAVING THE TRAINED MODEL
# The trained model is saved in a 'joblib file,
# and it will be later loaded in another code,

ml_filename = 'ML_samples'
trained_model = joblib.dump(model, ml_filename + ".joblib", compress=0)
