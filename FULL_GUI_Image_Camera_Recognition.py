# ------------------------------------ UNO CARDS RECOGNITION FROM IMAGE AND CAMERA STREAM
# ------------------------------------ WITH GRAPHICAL USER INTERFACE
# This code contain both Image and Camera Stream Recognition.
# A GUI has been added, so that selecting the recognition mode will result quicker and easier for the users

# ----------------------------------------------------- IMPORT LIBRARIES

import tkinter
from tkinter import Menu
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from itertools import repeat
import joblib

# ----------------------------------------------------- DEFINING HSV COLORS BOUNDARIES (MIN, MAX)
# Setting lower and higher hsv ranges for colors, used later to determine appropriate colors' masks
# The color red has two separare lower and upper ranges because it wraps around
# the circular hsv space (initial range, and final range)

# -------------------- HSV BLUE MIN AND MAX
b_hsv_min = np.array([87, 150, 80])
b_hsv_max = np.array([117, 255, 255])

# -------------------- HSV RED MIN AND MAX

r1_hsv_min = np.array([0, 100, 190])
r1_hsv_max = np.array([10, 255, 255])

r2_hsv_min = np.array([135, 85, 110])
r2_hsv_max = np.array([180, 255, 255])

# -------------------- HSV GREEN MIN AND MAX
g_hsv_min = np.array([40, 50, 70])
g_hsv_max = np.array([75, 255, 255])

# -------------------- HSV YELLOW MIN AND MAX
y_hsv_min = np.array([15, 170, 170])
y_hsv_max = np.array([35, 230, 255])

# ----------------------------------------------------- MAIN CODE

# Image_Recognition() function to call code for uno cards recognition from images
def Image_Recognition():
    
    # win_plot() function to display images
    def win_plot(image, im_title = 'bitmap_image'):
        cv2.imshow(im_title, image)
        cv2.waitKey(0)
        cv2.destroyWindow(im_title)
        return()
    
    # detect_color() function will detect the card's color and display the image with correspondent color string
    # a second parameter 'number' as been added, to later diaply both the detected color and the
    # number label estimated from the machine learning model
    def detect_color(card, number):
        card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

        # create colors' masks, defining boundaries for minimun and maximum hsv values
        blue_mask = cv2.inRange(card_hsv, b_hsv_min, b_hsv_max)
        red1_mask = cv2.inRange(card_hsv, r1_hsv_min, r1_hsv_max)
        red2_mask = cv2.inRange(card_hsv, r2_hsv_min, r2_hsv_max)
        green_mask = cv2.inRange(card_hsv, g_hsv_min, g_hsv_max)
        yellow_mask = cv2.inRange(card_hsv, y_hsv_min, y_hsv_max)

        # Dictionary storing color name (key) and hsv mask (value)
        colors = {'Blue' : blue_mask, 'Red' : red1_mask, 'Red' : red2_mask, 'Green' : green_mask, 'Yellow' : yellow_mask}

        # In this loop, the color of the card is determined.
        #Check the image pixels with each mask and returns binary values 0 and 1
        # If pixels of X_mask color is detected, and their total number is bigger than y,
        # Display the name of the color (dictionary key)
        for col, col_mask in enumerate(colors.values()):
            check_mask = np.count_nonzero(col_mask)  
            if np.sum(col_mask) > 0 and check_mask > 4000:
                color_name = str(list(colors.keys())[col])
                number = str(number)
                cv2.putText(card, color_name + number, org, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
                win_plot(card, 'UNO Card Recognition')
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

    filenames = glob.glob('C:/Users/elena/Desktop/UNO_Cards/Dataset/*.jpg')
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
    load_model = joblib.load('C:\\Users\\elena\\Desktop\\University Stuff\\3rd Year\\PDE-3433 Advanced Robotics\\Uno_Cards\\ML_samples.joblib')

    #Print out theaccuracy percentage (0-1)
    print(load_model.score(X_test, y_test))
    #print(load_model.predict(X_test))
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

                
# Camera_Recognition() function to call code for uno cards recognition from video stream       
def Camera_Recognition():
    #display_color() will check each color mask stored in 'colors' dictionary
    # and display the label of the correspondent color (dictionary key)
    def display_color(dictionary, numbr):
        for key in dictionary:
            contours, hierarchy = cv2.findContours(dictionary[key], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
             area = cv2.contourArea(contour)
             if(area > 0.5):
                numbr = str(numbr)
                cv2.putText(frame, key + numbr, text_space, cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0))

# ----------------------------- ASSIGN VARIABLES
    model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    camera = cv2.VideoCapture(0)
    kernel = np.ones((5,5), np.uint8)
    text_space = (10,30)

# --------------------------- MAIN - CAMERA STREAM INITIATION
    while camera.isOpened():
        rval, frame = camera.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # sensitivity, hsv min/max to detect contours by card white spaces
        sensitivity =  55
        hsv_min = np.array([0,0,255-sensitivity])
        hsv_max = np.array([255,sensitivity,255])

        frame_hsv_th = cv2.inRange(hsv_frame, hsv_min, hsv_max)
        # creating masks for colors (minimum and maximum hsv range)
        blue_mask = cv2.inRange(hsv_frame, b_hsv_min, b_hsv_max)
        red1_mask = cv2.inRange(hsv_frame, r1_hsv_min, r1_hsv_max)
        red2_mask = cv2.inRange(hsv_frame, r2_hsv_min, r2_hsv_max)
        green_mask = cv2.inRange(hsv_frame, g_hsv_min, g_hsv_max)
        yellow_mask = cv2.inRange(hsv_frame, y_hsv_min, y_hsv_max)
        # Dictionary storing color label (key) and respective color mask (value)
        colors = {'Blue' : blue_mask, 'Red' : red1_mask, 'Green' : green_mask, 'Yellow' : yellow_mask}

        frame_col_close = cv2.morphologyEx(frame_hsv_th, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(frame_col_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Extract features for contour corrisponding to central, larger number
# Loop through all contours and calculate their perimeter.
# If the perimeter is between x and y (hence, select number contour)
# obtain fuatures for Machine learning (Perimeter, Area, Corners, Axes  ratio, Convexity lenght)
# and store them in the list 'Geometry'
        for i,c in enumerate(contours):
            perimeter = cv2.arcLength(c, True)
            if perimeter >280 and perimeter < 600:
                area = cv2.contourArea(c)
                epsilon = 0.02*perimeter
                vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))
                ellipse = cv2.fitEllipse(c)
                (xc,yc),(d1,d2),angle = ellipse
                axes_ratio = round(d1/d2, 3)
                convexHull = cv2.convexHull(c)
                CH_len = len(convexHull)
                geometry = [[perimeter, area, vertex_approx, axes_ratio, CH_len]]
                geometry = np.array(geometry)
                # Draw contours on the card (optional, but recommended for better card positioning)
                #cv2.drawContours(frame, contours, -1, (0,0,0), 1)
                cv2.ellipse(frame, ellipse, (0, 0, 0), 2)
                # Load trained model with joblib, predict outcome, and display it along with color detection
                saved_model = joblib.load('C:\\Users\\elena\\Desktop\\University Stuff\\3rd Year\\PDE-3433 Advanced Robotics\\Uno_Cards\\ML_samples.joblib')
                predict_num = saved_model.predict(geometry)
                display_color(colors, predict_num)

        cv2.imshow("Camera Recognition", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    camera.release()

# ----------------------------------------------------- TKINTER CONFIGURATION

window = tkinter.Tk()
window.geometry('520x300')
window.configure(bg = 'black')
window.title('Assessment 2 - Computer Vision')

# Create button to access code for uno cards recognition from images
button1 = Button(window, text = ('Image Recognition'), fg = 'white',bd = 5, bg = 'black', font ='Times 20 bold', command = Image_Recognition)
button1.place(x = 1, y = 240)

# Create button to access code for uno cards recognition from camera stream
button2 = Button(window, text = ('Camera Recognition'), fg = 'white',bd = 5, bg = 'black', font ='Times 20 bold', command = Camera_Recognition)
button2.place(x = 250, y = 240)

label1 = Label(window, text = 'UNO Cards Recognition', font =('Times 25 italic bold'), fg = '#1ACE0E', bg = 'black')
label1.pack(side = TOP)

# Add picture to main GUI
tkframe = Frame(window)
tkframe.pack()
tkframe.place(anchor = 'center', relx=0.5, rely=0.47)
pic = ImageTk.PhotoImage(Image.open("C:/Users/elena/Desktop/uno_cards1.jpg"))
label2 = Label(tkframe, image = pic)
label2.pack()

# Create menu bar with File cascade window
# Add Exit option to File cascade to terminate the code
menubar = Menu(window)
window.config(menu = menubar)
file_menu = Menu(menubar)
file_menu.add_command(label = 'Exit', command = window.destroy)
menubar.add_cascade( label = 'File', menu = file_menu, underline = 0)

