import cv2 # type: ignore
import numpy as np # type: ignore

class LineDetector:
    def __init__(self, lower_yellow, upper_yellow):
        self.lower_yellow = np.array(lower_yellow)
        self.upper_yellow = np.array(upper_yellow)
        self.kernel = np.ones((5, 5), np.uint8)

    def detect_line(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2] if len(contours_result) == 3 else contours_result[0]
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 3)
                return frame, cx
        
        return frame, None