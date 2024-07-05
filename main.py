# main.py
import cv2
from color_detector import ColorDetector

def main():
    detector = ColorDetector()
    while True:
        detector.detect_colors()
        print("Detected labels:", detector.get_detected_labels())
        # name = str(detector.get_detected_labels())
        # print(type(name))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    detector.close_windows()

if __name__ == "__main__":
    main()
