import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2 # type: ignore
class RobotController:
    def __init__(self, line_detector, speed_controller):
        rospy.init_node('line_follower_robot')
        
        self.bridge = CvBridge()
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        
        self.line_detector = line_detector
        self.speed_controller = speed_controller

    def image_callback(self, image_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            frame = cv2.resize(frame, (640, 480))
            
            line_center = self.line_detector.detect_line(frame)
            frame_width = frame.shape[1]
            
            twist = Twist()
            if line_center is not None:
                twist.linear.x = self.speed_controller.adjust_linear_speed(line_center, frame_width)
                twist.angular.z = self.speed_controller.calculate_angular_velocity(line_center, frame_width)
            else:
                twist.linear.x = 0
                twist.angular.z = self.speed_controller.angular_speed
            
            self.cmd_vel_pub.publish(twist)
            cv2.imshow("Line Detection", frame)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("Error in image callback: %s", str(e))

    def run(self):
        rospy.loginfo("Robot controller running...")
        rospy.spin()
