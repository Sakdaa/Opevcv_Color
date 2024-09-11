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
img = cv2.VideoCapture(0)
img.set(3, 640)
img.set(4, 480)
# img = cv2.imread("1.jpg") #กีวี

# img = cv2.resize(img, (640, 480))
kernel = np.ones((5, 5), "uint8")
while True:
    ret, imageFrame = img.read()
    imageFrame = cv2.rotate(imageFrame, cv2.ROTATE_180)
    if not ret:
        break
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
    hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only the selected colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
            # Creating contour to track color

    upper_bound = np.array([blue_up, green_up, red_up])
    print(f"lower_upper: {lower_bound,blue_up,green_up,red_up}")
    # color_mask = cv2.dilate(mask, kernel)
    # res_color = cv2.bitwise_and(imageFrame, imageFrame, mask=color_mask)
    # contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     print(f"Detected labels: {area}")

    # Display the original image with contours and the mask
    cv2.imshow("Image with Contours", imageFrame)
    cv2.imshow("Color Trackbar", mask)
    
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break


cv2.destroyAllWindows()
