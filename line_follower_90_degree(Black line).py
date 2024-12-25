#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import tf  # for transforming IMU quaternion data to Euler angles
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import abc


# Abstract Interfaces
class IMU_Interface(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def update_imu(self, imu_msg):
        pass

    @abc.abstractmethod
    def get_yaw(self):
        pass


class Camera_Interface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update_image(self, img_msg):
        pass

    @abc.abstractmethod
    def get_processed_mask(self):
        pass

    @abc.abstractmethod
    def detect_centroids(self, mask):
        pass

    @abc.abstractmethod
    def show_camera_feed(self, mask=None):
        pass


class Motion_Interface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def control_move(self, linear_x, angular_z):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def follow_line(self, centroids, width):
        pass

    @abc.abstractmethod
    def turn_90_degrees(self, imu_reader, direction="left"):
        pass

    @abc.abstractmethod
    def move_forward(self, distance):
        pass



class IMU_Handler(IMU_Interface):
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
        self.yaw = (self.yaw * 180 / np.pi) % 360  # convert yaw to degrees

    def get_yaw(self):
        """
        Returns the current yaw angle
        """
        return self.yaw


class Camera_Processor(Camera_Interface):
    """
    Processes camera images to detect black lines
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

    def update_image(self, img_msg):
        """
        Callback  function to update the current frame with incoming camera images

        Args:
            img_msg (Image): The ROS Image message containing the camera feed
        """
        try:
            self.frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logerr("Failed to convert image: {}".format(e))

    def get_processed_mask(self):
        """
        Converts the image to HSV and creates a mask for detecting black lines

        Returns:
            np.array or None:Binary mask of the region of interest (ROI) if the frame is available otherwise None
        """
        if self.frame is None:
            return None

        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        
        #changable depending on color of the line
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        mask_hsv = cv2.inRange(hsv, lower_black, upper_black)

        kernel = np.ones((5, 5), np.uint8)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)

        height, width = mask_hsv.shape
        roi = mask_hsv[int(height * 0.7):, :]
        return roi

    def detect_centroids(self, mask):
        """
        Detects centroids of the yellow line in the mask

        Args:
            mask (np.array): Binary mask of the image

        Returns:
            list: List of detected centroids or None values
            int: Width of the mask
        """
        if mask is None:
            return [], 0
        height, width = mask.shape
        slices = [mask[int(height * ratio):int(height * ratio) + 10, :] for ratio in [0.8, 0.6, 0.4]]
        moments_list = [cv2.moments(s) for s in slices]
        centroids = [
            (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) if m["m00"] > 0 else None
            for m in moments_list
        ]
        return centroids, width

    def show_camera_feed(self, mask=None):
        """
        Displays the current camera feed and an optional mask for visualizing detections

        Args:
            mask (np.ndarray): A binary or grayscale mask to display(Default is None)
        """
        if mask is not None:
            cv2.imshow("Mask", mask)
        if self.frame is not None:
            cv2.imshow("Camera Feed", self.frame)
            cv2.waitKey(1)


class Motion_Handler(Motion_Interface):
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

    def control_move(self, linear_x, angular_z):
        """
        Send movement commands to the robot

        Args:
            linear_x (float): Forward velocity
            angular_z (float): Angular velocity(turning rate)
        """
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def stop(self):
        """
        Stop the robot by setting velocities to zero.
        """
        self.control_move(0.0, 0.0)

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
            self.control_move(0.4, -float(error) / 100)  # speed and turning rate based on centroid error
            return True
        else:
            self.stop() # stop if no line is detected
            return False

    def turn_90_degrees(self, imu_reader, direction="left"):
        """
        Turn the robot 90 degrees in a specified direction.

        Args:
            imu_reader (IMUInterface): IMU handler for yaw data.
            direction (str): Direction to turn ("left" or "right").
        """
        target_angle = (imu_reader.get_yaw() + (90 if direction == "left" else -90)) % 360
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            current_yaw = imu_reader.get_yaw()
            error = (target_angle - current_yaw + 360) % 360
            if error > 180:
                error -= 360
            if abs(error) < 2:  # stop when close enough(adjustable)
                self.stop()
                break
            angular_z = 0.5 if error > 0 else -0.5
            self.control_move(0.0, angular_z)
            rate.sleep()

    def move_forward(self, distance):
        """
        Move the robot forward for a specified distance.

        Args:
            distance (float): Distance to move forward (time duration in seconds).
        """
        self.control_move(0.2, 0.0)
        rospy.sleep(distance)  
        self.stop()


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
        self.line_lost_counter = 0

    def run(self):
        """
        Main method that will be called to start the process
        """
        rate = rospy.Rate(10)
        state = "FOLLOW_LINE"   #handle the twon states FOLLOW_LINE or TURN

        while not rospy.is_shutdown():
            mask = self.camera_processor.get_processed_mask()
            centroids, width = self.camera_processor.detect_centroids(mask)

            # show camera feed and mask
            self.camera_processor.show_camera_feed(mask=mask)

            if state == "FOLLOW_LINE":
                if any(centroids):
                    self.line_lost_counter = 0
                    self.motion_controller.follow_line(centroids, width)
                else:
                    self.line_lost_counter += 1
                    if self.line_lost_counter > 10:  # lost line for a while(adjustable)
                        rospy.loginfo("Line lost. Preparing to turn...")
                        state = "TURN"

            elif state == "TURN":
                rospy.loginfo("Moving forward to detect new line direction...")
                self.motion_controller.move_forward(1)  # move forward for 1 second (adjustable)

                # re-evaluate centroids to determine turn direction
                mask = self.camera_processor.get_processed_mask()
                centroids, width = self.camera_processor.detect_centroids(mask)

                #detects the direction of turning
                if any(centroids):
                    cx = next((c[0] for c in centroids if c is not None), width // 2)
                    if cx < width // 2:
                        turn_direction = "left"
                    else:
                        turn_direction = "right"

                    self.motion_controller.turn_90_degrees(self.imu_reader, direction=turn_direction)
                else:
                    rospy.logwarn("No line detected after moving forward. Defaulting to left turn...")
                    self.motion_controller.turn_90_degrees(self.imu_reader, direction="left")

                state = "FOLLOW_LINE"

            rate.sleep()


#main
if __name__ == "__main__":
    rospy.init_node("line_follower")
    result = Line_Follower_System()
    result.run()
