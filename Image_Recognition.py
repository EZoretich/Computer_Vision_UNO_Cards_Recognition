# ------------------------------------------  UNO CARD RECOGNITION FROM IMAGES
# --------------------- IMPORT LIBRARIES
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from itertools import repeat
import joblib

# ---------------------  DEFINE FUNCTIONS
#win_plot() is used to display images
def win_plot(image, im_title = 'bitmap_image'):
    cv2.imshow(im_title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(im_title)
    return()

# crop() was initially used to crop the images and display the card of interest without the background
# although, in order to visualize better the displaied result of machine learning predictions,
# crop() function has been set aside
'''def crop(card):
    card_bw = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    card_smooth = cv2.blur(card_bw, (5, 5))
    thr_value, card_th = cv2.threshold(card_smooth, 150, 255, cv2.THRESH_BINARY)
    card_close = cv2.morphologyEx(card_th, cv2.MORPH_CLOSE, kernel)
    card_canny = cv2.Canny(card_close, 50, 100)
    contours, hierarchy = cv2.findContours(card_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    [x,y,w,h] = cv2.boundingRect(contours[0])
    card_crop = card[y:y+h,x:x+w,:]
    return card_crop'''

#detect_color() uses colors' masks and numpy build it functions, to determine
# the color of the displayed card
def detect_color(card, number):
    #card = crop(card)
    
    # Setting lower and higher hsv ranges for colors, used later to determine appropriate range of colors' masks
    # The color red has two separare lower and upper ranges because it wraps around
    # the circular hsv space (initial range, and final range)
    # -------------------- HSV BLUE MIN AND MAX
    b_hsv_min = np.array([50, 110, 150])
    b_hsv_max = np.array([130, 255, 255])

    # -------------------- HSV RED MIN AND MAX
    r_hsv_min = np.array([0, 100, 190])
    r_hsv_max = np.array([10, 255, 255])

    # -------------------- HSV GREEN MIN AND MAX
    g_hsv_min = np.array([40, 50, 70])
    g_hsv_max = np.array([100, 255, 255])

    # -------------------- HSV YELLOW MIN AND MAX
    y_hsv_min = np.array([15, 170, 190])
    y_hsv_max = np.array([50, 230, 255])

    card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

    # Creating masks for colors
    blue_mask = cv2.inRange(card_hsv, b_hsv_min, b_hsv_max)
    red_mask = cv2.inRange(card_hsv, r_hsv_min, r_hsv_max)
    green_mask = cv2.inRange(card_hsv, g_hsv_min, g_hsv_max)
    yellow_mask = cv2.inRange(card_hsv, y_hsv_min, y_hsv_max)
    
    # Dictionary storing color name (key) and hsv mask (value)
    colors = {'Blue' : blue_mask, 'Red' : red_mask, 'Green' : green_mask, 'Yellow' : yellow_mask}

    # In this loop, the color of the card is determined.
    #Check the image pixels with each mask and returns binary values 0 and 1
    # If pixels of X_mask color is detected, and their total number is bigger than y,
    # Display the name of the color (dictionary key)
    for col, col_mask in enumerate(colors.values()):
        check_mask = np.count_nonzero(col_mask)  
        if np.sum(col_mask) > 0 and check_mask > 3000:
            color_name = str(list(colors.keys())[col])
            number = str(number)
            cv2.putText(card, color_name + number, org, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            win_plot(card, 'Uno Card Recognition')
            break

# obt_features() function trace the contours of the card (following the white spaces),
# detect the contours of the central, larger number, and obtain the necessary
# features to later on train the Machine Learning model
def obt_features(card):
    # sensitivity, hsv min/max to detect contours by card white spaces
    sensitivity =  55
    hsv_min = np.array([0,0,255-sensitivity])
    hsv_max = np.array([255,sensitivity,255])

    card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    card_hsv_th = cv2.inRange(card_hsv, hsv_min, hsv_max)

    card_col_close = cv2.morphologyEx(card_hsv_th, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(card_col_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Loop through all contours and calculate their perimeter.
    # If the perimeter is between x and y (hence, select number contour)
    # obtain fuatures for Machine learning (Perimeter, Area, Corners, Axes  ratio, Convexity lenght)
    # and store them in the list 'Geometry'
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
            geometry = np.array(geometry)
            return geometry

# ----------------------------- ASSIGN VARIABLES
org = (10,30)
kernel = np.ones((3,3), np.uint8)
ml_data = []
labels = []
model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
cards_path = 'C:/Users/elena/Desktop/UNO_Cards/Dataset/*.jpg'
model_path = 'C:\\Users\\elena\\Desktop\\University Stuff\\3rd Year\\PDE-3433 Advanced Robotics\\Uno_Cards\\ML_samples.joblib'

# ----------------------------- ADDING CARD LABELS
# The below loops will create a list of numbers corrisponding to the label of each uno card
# The initial assignment of those label has been made from the folder in which all 280 cards
# were stored. The folder was sorted by Name --> Ascending
for l1 in range(4):
    for l2 in range(10):
        labels.append(l2)
for l3 in range(4):
    for l4 in range(0,10):
        labels.extend(repeat(l4, 6))

# --------------------------- LOOPING THROUGH ALL IMAGES
# Call function for obtaining features throughout all cards,
# Then add the array of selected features into another array (ml_data)
        
filenames = glob.glob(cards_path)
images = [cv2.imread(img) for img in filenames]
for img in images:
    D = obt_features(img)
    ml_data.append(D)

ml_data = np.array(ml_data)

X = ml_data
y = labels

# Split features and labels into training and testing sets.
# ±86% of cards(features) will be used for training, and teh remaining 14% will be used for testing
# train_test_split() argument 'shuffle' has been set to True, allowing a random selection of card
# to be used each time for training and testing. This will create a non-bias model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.14, shuffle=True)

# Loading the Machine Learning trained model, with joblib
load_model = joblib.load(model_path)

#Print out theaccuracy percentage (0-1)
print(load_model.score(X_test, y_test))
prediction = load_model.predict(X_test)


# The following loops allow to display the result from machine learning number estimation
# in the respective card image
j = 0
for element in X_test:
    j += 1
    for img in images:
        x = obt_features(img)
        if (element == x).all():
            detect_color(img, prediction[j-1])

