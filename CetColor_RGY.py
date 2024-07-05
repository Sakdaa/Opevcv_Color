import cv2
import numpy as np

# Initialize the trackbar window
cv2.namedWindow("Color Trackbar")
cv2.namedWindow("Image with Contours")

def display(value):
    pass

# Create trackbars for upper and lower color range
cv2.createTrackbar("B_up", "Color Trackbar", 0, 255, display)
cv2.createTrackbar("G_up", "Color Trackbar", 0, 255, display)
cv2.createTrackbar("R_up", "Color Trackbar", 0, 255, display)

cv2.createTrackbar("B_low", "Color Trackbar", 0, 255, display)
cv2.createTrackbar("G_low", "Color Trackbar", 0, 255, display)
cv2.createTrackbar("R_low", "Color Trackbar", 0, 255, display)

# Load the image
img = cv2.imread("1.jpg") #กีวี
img = cv2.resize(img, (640, 480))

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = img[y, x]
        print(f"RGB Value at ({x}, {y}): {pixel}")
        cv2.setTrackbarPos("B_low", "Color Trackbar", max(0, pixel[0] - 20))
        cv2.setTrackbarPos("G_low", "Color Trackbar", max(0, pixel[1] - 20))
        cv2.setTrackbarPos("R_low", "Color Trackbar", max(0, pixel[2] - 20))
        cv2.setTrackbarPos("B_up", "Color Trackbar", min(255, pixel[0] + 20))
        cv2.setTrackbarPos("G_up", "Color Trackbar", min(255, pixel[1] + 20))
        cv2.setTrackbarPos("R_up", "Color Trackbar", min(255, pixel[2] + 20))

cv2.setMouseCallback("Image with Contours", mouse_callback)

while True:
    # Get values from trackbars
    blue_up = cv2.getTrackbarPos("B_up", "Color Trackbar")
    green_up = cv2.getTrackbarPos("G_up", "Color Trackbar")
    red_up = cv2.getTrackbarPos("R_up", "Color Trackbar")

    blue_low = cv2.getTrackbarPos("B_low", "Color Trackbar")
    green_low = cv2.getTrackbarPos("G_low", "Color Trackbar")
    red_low = cv2.getTrackbarPos("R_low", "Color Trackbar")

    # Define the range of the color in HSV
    lower_bound = np.array([blue_low, green_low, red_low])
    upper_bound = np.array([blue_up, green_up, red_up])

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only the selected colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Display the original image with contours and the mask
    cv2.imshow("Image with Contours", img)
    cv2.imshow("Color Trackbar", mask)
    
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

cv2.destroyAllWindows()
