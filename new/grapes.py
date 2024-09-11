import numpy as np
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from networktables import NetworkTables
from networktables.util import ntproperty
import threading
import time
import os



#Create thread to make sure networktables is connected
cond = threading.Condition()
notified = [False]

#Create a listener
def connectionListener(connected, info):
    with cond:
        notified[0] = True
        cond.notify()
#Instantiate NetworkTables
NetworkTables.initialize(server="10.20.25.2")
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)


#Create the vision Table
ntStopBarcode = ntproperty('/Vision/stopBarcode', False)
#Get Table
table = NetworkTables.getTable('Vision')

# Capturing video through webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)

# Define color ranges and corresponding labels
color_ranges = [
    ("Red", [145, 65, 11], [186, 201, 169], 4000, 40000),#[0, 0, 0], [24, 255, 30], 2000, 90000)
    ("green", [24, 56, 81], [35, 148, 181], 4000, 40000),#"green", [4, 12, 122], [54, 127, 172], 2000, 90000),
    ("yellow", [16, 40, 71], [22, 176, 255], 4000, 50000), # [16, 98, 109], [23, 194, 200], 4000, 30000)#"yellow", [2, 57, 178], [36, 128, 255], 3000, 90000), 2000, 90000) #"yellow", [13, 92, 127], [31, 255, 216], 2000, 90000)
]

# Function to detect and label fruits based on their color and area
def detect_and_label_fruits(imageFrame, hsvFrame, color_ranges, kernel):
    detected_fruits = []
    for label, lower, upper, area_ref_low, area_ref_up in color_ranges:
        lower = np.array(lower, np.uint8)
        upper = np.array(upper, np.uint8)

        # Create a mask for the color range
        color_mask = cv2.inRange(hsvFrame, lower, upper)
        color_mask = cv2.dilate(color_mask, kernel)
        res_color = cv2.bitwise_and(imageFrame, imageFrame, mask=color_mask)

        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area_ref_low < area < area_ref_up:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2 
                if w <= 220:
                    cv2.drawContours(roi, [contour], -1, (0, 255, 0), 3)
                    cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(imageFrame, label, (cx-15, cy+65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                    detected_fruits.append((label, (cx, cy)))
                #print(cx,cy)
    return detected_fruits



def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = contrast / 127.0 + 1.0
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# Kernel for morphological transformation
kernel = np.ones((5, 5), "uint8")

while True:  
    # Reading the video from the webcam in image frames
    ret, imageFrame = webcam.read()
    imageFrame = cv2.rotate(imageFrame, cv2.ROTATE_180)
    
    if not ret:
        break

    # Define region of interest (imageFrame) - lower half of the frame
    height, width, _ = imageFrame.shape
    roi = imageFrame[0:height-60, 0:width-180]
    #roi = imageFrame[130:height, 0:480]

    # Convert the imageFrame from BGR (RGB color space) to HSV color space
    hsvFrame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    detected_fruits = detect_and_label_fruits(roi, hsvFrame, color_ranges, kernel)

    # Log detected fruits and their positions '/home/pi/Camera/detected_grapes.txt'
    
    with open('detected_grapes.txt', 'w') as f:
        for fruit, (cx, cy) in detected_fruits:
            f.write(f'{fruit}\n')
            f.write(f'{cx}\n')
            f.write(f'{cy}\n')


    #cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)
    cv2.imshow("Region of Interest", roi)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break 
    # table.getBoolean('stopBarcode', False) == True:
        #table.putBoolean('stopBarcode', False)
        #break     

# Release the capture and close windows
webcam.release()
cv2.destroyAllWindows()
