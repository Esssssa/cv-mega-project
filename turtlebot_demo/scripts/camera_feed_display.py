#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def display_camera_feed():
    rospy.init_node('camera_feed_display')
    bridge = CvBridge()

    def callback(image_msg):
        try:
            frame = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)
        except Exception as e:
            # rospy.logerr(f"Error: {e}")
            pass

    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)
    rospy.loginfo("Displaying Camera Feed...")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        display_camera_feed()
    except rospy.ROSInterruptException:
        pass