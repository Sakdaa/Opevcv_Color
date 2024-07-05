# color_detector.py

import numpy as np
import cv2

class ColorDetector:
    def __init__(self, webcam_index=1, width=640, height=480):
        # Initialize webcam
        self.webcam = cv2.VideoCapture(webcam_index)
        self.webcam.set(3, width)
        self.webcam.set(4, height)
        
        # Define color ranges and corresponding labels
        self.color_ranges = [
            ("R", [155, 35, 38], [177, 245, 216], 13000, 90000),
            ("G", [0, 30, 155], [25, 150, 248], 4000, 10000),
            ("Y", [13, 0, 278], [32, 141, 255], 20000, 90000),
        ]
        
        # Kernel for morphological transformation
        self.kernel = np.ones((5, 5), "uint8")
        
        # Store detected labels
        self.detected_labels = []

    def detect_colors(self):
        # Reading the video from the webcam in image frames
        ret, imageFrame = self.webcam.read()
        #roi = imageFrame[500:720, 0:1280] #cut imageFrame

        # Convert the imageFrame in BGR (RGB color space) to HSV color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        for label, lower, upper, area_ref_low, area_ref_up in self.color_ranges:
            # Set range for the color and define mask
            lower = np.array(lower, np.uint8)
            upper = np.array(upper, np.uint8)
            color_mask = cv2.inRange(hsvFrame, lower, upper)

            # Morphological Transform, Dilation
            color_mask = cv2.dilate(color_mask, self.kernel)
            res_color = cv2.bitwise_and(imageFrame, imageFrame, mask=color_mask)

            # Creating contour to track color
            contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area_ref_low < area < area_ref_up:
                    self.detected_labels.clear()
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(imageFrame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
                    self.detected_labels.append(label)

        # Display the result
        cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)


    def close_windows(self):
        # Release the capture and close windows
        self.webcam.release()
        cv2.destroyAllWindows()

    def get_detected_labels(self):
        # Return unique detected labels
        return list(set(self.detected_labels))

    def save_detected_labels(self, filename="detected_labels.txt"):
        unique_labels = self.get_detected_labels()
        with open(filename, "w") as file:
            for label in unique_labels:
                file.write(label + "\n")
