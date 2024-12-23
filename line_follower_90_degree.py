#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import tf               #for transforming IMU quaternion data to Euler angles
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class IMU_Handler:
    """
    Reads and processes IMU data to extract the yaw angle
    """
    def __init__(self, topic="/imu"):
        """
        Initializes the IMU handler and subscribes to the correct topic

        Args:
            topic (str): The ROS topic to subscribe to for IMU data
        """
        self.yaw = 0.0
        rospy.Subscriber(topic, Imu, self.update_imu)

    def update_imu(self, imu_msg):
        """
        Callback function that is called whenever new yaw data is received from the ROS topic
        
        Args:
            imu_msg (Float64): The message containing the new yaw value
        """
        orientation = imu_msg.orientation
        _, _, self.yaw = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.yaw = (self.yaw * 180 / np.pi) % 360  # convert yaw angle to degrees

    def get_yaw(self):
        """
        Returns the current yaw angle
        """
        return self.yaw


class Camera_Processor:
    """
    Processes camera images to detect yellow lines 
    """
    def __init__(self, topic="/camera/rgb/image_raw"):
        """
        Initializes the camera processor and subscribes to the coorect topic 

        Args:
            topic (str): The ROS topic to subscribe to for camera data
        """
        self.bridge = CvBridge()
        rospy.Subscriber(topic, Image, self.update_image)
        self.frame = None
        self.processed_frame = None

    def update_image(self, img_msg):
        """
        Callback  function to update the current frame with incoming camera images

        Args:
            img_msg (Image): The ROS Image message containing the camera feed
        """
        try:
            self.frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logerr("Failed to convert image: {}".format(e))  # compaitable with python2

    def get_processed_mask(self):
        """
        Converts the image to HSV and creates a mask for detecting yellow lines

        Returns:
            np.array or None: Binary mask or None if no frame is available.
        """
        if self.frame is None:
            return None
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        
        # define HSV range for yellow color(changable depending on the line color)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        #to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def detect_centroids(self, mask):
        """
        Detects centroids of the yellow line in the mask

        Args:
            mask (np.array): Binary mask of the image

        Returns:
            list: List of detected centroids or None values
            int: Width of the mask
        """
        if mask is None:  #hanadle empty mask or error cases
            return [], 0
        height, width = mask.shape

        
        #handle all versions of openVC
        try:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            self.processed_frame = self.frame.copy()
            cv2.drawContours(self.processed_frame, contours, -1, (0, 255, 0), 2) # Draw contours on the frame for visualization
        else:
            self.processed_frame = self.frame

        # Slice the mask to detect centroids at specific heights
        slices = [mask[int(height * ratio):int(height * ratio) + 5, :] for ratio in [0.7, 0.5, 0.3]]
        moments_list = [cv2.moments(s) for s in slices]
        centroids = [(int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])) if m['m00'] > 0 else None for m in moments_list]

        return centroids, width


    def show_camera_feed(self):
        """
        Shows the current camera frame with detection
        """
        if self.processed_frame is not None:
            cv2.imshow("Camera Feed with Detection", self.processed_frame)
            cv2.waitKey(1)                   
        elif self.frame is not None:
            cv2.imshow("Camera Feed", self.frame)
            cv2.waitKey(1)





class Motion_Handler:
    """
    Controls the robot's motion
    """
    def __init__(self, topic="/cmd_vel"):
        """
        Initializes the motion controller with a topic
        
        Args:
            topic (str): The ROS topic for publishing movement commands
        """
        self.cmd_pub = rospy.Publisher(topic, Twist, queue_size=10)
        self.turning = False
        self.initial_yaw = None
        self.turn_direction = 0

    def follow_line(self, centroids, width):
        """
        Moves the robot to follow the detected line

        Args:
            centroids (list): List of detected centroids
            width (int): Width of the image mask
        Returns:
            bool: True if line is detected, otherwise False.
        """
        if any(centroids):
            cx = next((c[0] for c in centroids if c is not None), width // 2)
            error = cx - width // 2
            self.turn_direction = 1 if error < 0 else -1
            self.control_move(0.4, -float(error) / 100)  # speed and turn rate (can be changed based on prefrence)
            return True
        return False

    def initiate_turn(self, current_yaw):
        """
        Initiates the turning  when the line is lost
        
        Args:
            current_yaw (float): Current yaw angle from the IMU
        """
        self.turning = True
        self.initial_yaw = current_yaw
        self.control_move(0.0, 0.5 * self.turn_direction)

    def execute_turn(self, current_yaw):
        """
        Completes the turn when the bot has turned approximately 90 degrees
        
        Args:
            current_yaw (float): Current yaw angle from the IMU
        """
        yaw_diff = abs(current_yaw - self.initial_yaw)
        if yaw_diff >= 90:
            self.turning = False
            self.control_move(0.0, 0.0)

    def control_move(self, linear_x, angular_z):
        """
        Publishes motor commands to control the robot's movement
        
        Args:
            linear_x (float): Forward velocity
            angular_z (float): Angular velocity (turning rate)
        """
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)


class Line_Follower_System:
    """
    Main system handler
    """
    def __init__(self):
        """
        Initializes the components of the line-following system
        """
        self.imu_reader = IMU_Handler()
        self.camera_processor = Camera_Processor()
        self.motion_controller = Motion_Handler()

    def run(self):
        """
        Main method that will be called to start the process
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            mask = self.camera_processor.get_processed_mask()
            if mask is not None:
                centroids, width = self.camera_processor.detect_centroids(mask)
                current_yaw = self.imu_reader.get_yaw()

               
                self.camera_processor.show_camera_feed()  # Display the camera feed

                if not self.motion_controller.follow_line(centroids, width) and not self.motion_controller.turning:
                    rospy.loginfo("No line detected, initiating turn...")
                    self.motion_controller.initiate_turn(current_yaw)

                if self.motion_controller.turning:
                    self.motion_controller.execute_turn(current_yaw)

            rate.sleep()


#main
if __name__ == "__main__":
    rospy.init_node('line_follower')
    result = Line_Follower_System()
    result.run()
