#-------------------------------- UNO CARDS RECOGNITION FROM CAMERA STREAM

# ------------------------ IMPORT LIBRARIES
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib

# ------------------------ DEFINE FUNCTION
#display_color() will check each color mask stored in 'colors' dictionary and display the color
#(dictionary key)
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
while camera.isOpened():
    rval, frame = camera.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # sensitivity, hsv min/max to detect contours by card white spaces
    sensitivity =  55
    hsv_min = np.array([0,0,255-sensitivity])
    hsv_max = np.array([255,sensitivity,255])

    frame_hsv_th = cv2.inRange(hsv_frame, hsv_min, hsv_max)

    frame_col_close = cv2.morphologyEx(frame_hsv_th, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(frame_col_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # creating masks for colors (minimum and maximum hsv range)
    blue_mask = cv2.inRange(hsv_frame, b_hsv_min, b_hsv_max)
    red1_mask = cv2.inRange(hsv_frame, r1_hsv_min, r1_hsv_max)
    red2_mask = cv2.inRange(hsv_frame, r2_hsv_min, r2_hsv_max)
    green_mask = cv2.inRange(hsv_frame, g_hsv_min, g_hsv_max)
    yellow_mask = cv2.inRange(hsv_frame, y_hsv_min, y_hsv_max)
    # Dictionary storing color label (key) and respective color mask (value)
    colors = {'Blue' : blue_mask, 'Red' : red1_mask, 'Green' : green_mask, 'Yellow' : yellow_mask}

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
            cv2.drawContours(frame, contours, -1, (0,0,0), 1)
            # Draw ellipse around contours selected
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
