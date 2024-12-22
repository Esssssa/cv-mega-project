#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineFollower:
    def __init__(self):
        rospy.init_node('turtlebot_line_follower')
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        
        self.bridge = CvBridge()
        
        self.base_linear_speed = 0.2  # Normal linear speed
        self.min_linear_speed = 0.05  # Slowest linear speed near edges
        self.angular_speed = 0.5  # Max angular speed for turning
        
        self.slowdown_margin = 0.25  # Fraction of frame width near edges to slow down
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

    def detect_line(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
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

    def adjust_linear_speed(self, line_center, frame_width):
        """
        Dynamically adjust linear speed based on how close the line center is to the frame edges.
        """
        if line_center is None:
            return 0
        
        # Define edge slowdown regions
        edge_margin = int(self.slowdown_margin * frame_width)
        
        # Adjust linear speed based on proximity to edges
        if line_center < edge_margin or line_center > frame_width - edge_margin:
            # Slow down as target approaches the edges
            linear_speed = self.min_linear_speed
        else:
            # Gradually adjust speed based on distance from the center
            distance_from_center = abs(line_center - frame_width / 2)
            max_distance = frame_width / 2 - edge_margin
            speed_factor = 1 - (distance_from_center / max_distance)
            linear_speed = self.min_linear_speed + (self.base_linear_speed - self.min_linear_speed) * speed_factor
        
        return linear_speed

    def calculate_angular_velocity(self, line_center, frame_width):
        """
        Calculate angular velocity based on line position.
        """
        if line_center is None:
            return 0
        
        error = line_center - (frame_width / 2)
        angular_vel = -error * 0.005
        angular_vel = max(min(angular_vel, self.angular_speed), -self.angular_speed)
        return angular_vel

    def image_callback(self, image_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            frame = cv2.resize(frame, (640, 480))
            
            processed_frame, line_center = self.detect_line(frame)
            
            cv2.imshow("Line Detection", processed_frame)
            cv2.waitKey(1)
            
            twist = Twist()
            
            if line_center is not None:
                # Adjust linear and angular velocities dynamically
                twist.linear.x = self.adjust_linear_speed(line_center, frame.shape[1])
                twist.angular.z = self.calculate_angular_velocity(line_center, frame.shape[1])
            else:
                # No line detected, rotate in place to search
                twist.linear.x = 0
                twist.angular.z = self.angular_speed
            
            self.cmd_vel_pub.publish(twist)
            
        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

    def run(self):
        rospy.loginfo("Turtlebot Line Follower with Edge Slowdown Started")
        rospy.spin()

if __name__ == '__main__':
    try:
        line_follower = LineFollower()
        line_follower.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        cmd_vel_pub.publish(Twist())
        cv2.destroyAllWindows()
