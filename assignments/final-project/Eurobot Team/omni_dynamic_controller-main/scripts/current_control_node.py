#!/usr/bin/env python3
import warnings
import numpy as np
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

#rospy.logwarn("Hello world!")
# Define control rate
RATE_get_local_robot_speed = 100
RATE_publish_control = 50


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ControllerNode(object):
    """
    Controller operates in the local coordinate system assigned with the robot.
    Localization coordinates should be transformed into local robot coordinates.
    Target point should be transformed into local coordinate system.
    """

    def __init__(self):

        # Creating ROS Publishers
        self.pub_stm_command = rospy.Publisher("/secondary_robot/stm/command", String, queue_size=1)
        self.pub_wheel_speed = rospy.Publisher("/secondary_robot/wheel_speed", String, queue_size=1)
        self.pub_robot_command = rospy.Publisher("/secondary_robot/int_error", String, queue_size=1)

        # Creating ROS Subscribers
        rospy.Subscriber("/secondary_robot/stm/response", String, self.callback_response, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1. / RATE_get_local_robot_speed), self.tcallback_main_loop)
        self.timer_wheel_speed = rospy.Timer(rospy.Duration(1. / RATE_get_local_robot_speed), self.tcallback_wheel_speed)

    def tcallback_main_loop(self, event):
        pass

    def tcallback_wheel_speed(self, event):
        self.pub_stm_command.publish("0 7")

    def callback_response(self, data):
        """Function parses data from STM into robot local frame speed data.

        Each time when data is published in topic "/secondary_robot/stm/response",
        this function is called.

        :param data: Data from the STM topic "/secondary_robot/stm/response"
        :return: self.vx_local, self.vy_local, self.w_local
        """

        speed_vals = [float(token) for token in data.data.split()]
        int_error_string = "%f %f %f" % (speed_vals[1], speed_vals[2], speed_vals[3])
        rospy.logwarn("speed data: %s" % int_error_string)
        self.pub_robot_command.publish(int_error_string)

def shutdown():
    control_string = "0" + " 0x50 " + "0 0 0"
    controller_node.pub_stm_command.publish(control_string)
    rospy.logwarn("Controller shutting down")


if __name__ == '__main__':
    try:
        rospy.on_shutdown(shutdown)
        rospy.init_node('controller_node', anonymous=True)
        warnings.filterwarnings("error")
        controller_node = ControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
