#!/usr/bin/env python3
import warnings
import numpy as np
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
import tf2_ros

RATE_get_local_robot_speed = 50
RATE_publish_control = 50


def wrap_angle(angle):
    """
    Wraps the given angle to the range [-pi, +pi].
    :param angle: The angle (in rad) to wrap (can be unbounded).
    :return: The wrapped angle (guaranteed to in [-pi, +pi]).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unfold_angles(angles):
    """
    Unfolds the angle array to be >= +- pi
    :param angle: An array of angles.
    :return: Array of unwraped angles.
    """
    angles = wrap_angle(angles)
    delta = angles[1:] - angles[:-1]
    delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)
    delta = np.where(delta < -np.pi, delta + 2 * np.pi, delta)
    return angles[0] + np.concatenate([np.zeros(1), np.cumsum(delta)], axis=0)


class ControllerNode(object):

    def __init__(self):
        self.robot_state = np.zeros(6)
        self.angle_array = []  # Angle array to unfold angle
        self.flag = False  # End of path flag

        # tf listener for localization
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # Creating path vosmerka
        angle = np.linspace(np.pi, 3 * np.pi, 150)
        x = (0.5 * 2 ** 0.5 * np.cos(angle)) / (1 + np.sin(angle) ** 2) + 1.5
        y = (0.5 * 2 ** 0.5 * np.cos(angle) * np.sin(angle)) / (1 + np.sin(angle) ** 2)
        self.path = np.array([x, y, angle]).T
        self.path = np.append(self.path, [[self.path[0, 0], self.path[0, 1], self.path[-1, 2]]], axis=0)
        self.path = np.append(self.path, [[0, 0, self.path[-1, 2]]], axis=0)

        self.current_goal = self.path[0]
        self.cnt = 0

        # Creating ROS Publishers
        self.pub_stm_command = rospy.Publisher("/secondary_robot/stm/command", String, queue_size=1)
        # Publisher for logger
        self.pub_robot_command = rospy.Publisher("/secondary_robot/robot_command", String, queue_size=1)
        self.state_pub_header = rospy.Publisher("/secondary_robot/robot_state_header", Header, queue_size=1)

        # Creating ROS Subscribers
        rospy.Subscriber("/secondary_robot/stm/response", String, self.callback_response, queue_size=1)

        # Initialize poll timers
        self.timer = rospy.Timer(rospy.Duration(1. / RATE_get_local_robot_speed), self.tcallback_speed)
        self.timer_control = rospy.Timer(rospy.Duration(1. / RATE_publish_control), self.tcallback_control, reset=True)

    def tcallback_control(self, event):
        """Function publishes control current signal

        Each time the timer ticks with requested rate,
        this function:
            1) Calculates state of the robot X: [x, y, theta, vx, vy, w]
            2) It publishes robot state X
            3) Calculates Control and gets summary control vector
            4) Publishes control command for the STM
        """

        pose_dev = self.calc_errors()
        if not self.flag:
            if self.is_near_goal():
                # control_string = "0" + " 0x08 " + "0 0 0"
                # rospy.logwarn("got to goal")
                # self.pub_stm_command.publish(control_string)

                if self.cnt < self.path.shape[0] - 1:
                    self.cnt += 1
                    self.current_goal = self.path[self.cnt]
                else:
                    self.flag = True
            else:
                state = Header()
                state.stamp = rospy.Time.now()

                state.frame_id = "%s %s %s %s %s %s" % (
                    self.robot_state[0], self.robot_state[1], self.robot_state[2],
                    self.robot_state[3], self.robot_state[4], self.robot_state[5])

                self.state_pub_header.publish(state)
                control_signal = self.get_control_signal(pose_dev)

                # Publish data to STM
                control_string = "%f %f %f" % (control_signal[0], control_signal[1], control_signal[2])
                control_string = "0" + " 0x08 " + control_string
                self.pub_stm_command.publish(control_string)
                self.pub_robot_command.publish(control_string)

        else:
            control_string = "0" + " 0x08 " + "0 0 0"
            self.pub_stm_command.publish(control_string)
            self.flag = True

    def get_control_signal(self, deviations):
        """
        Uses coefficents of the LQR controller to calculate the
        control vector for the omni-directional platform
        :param deviations: Array of deviations [x_dev, y_dev, theta_dev] from robot position to goal position.
        :return: Array of control outputs [control_signal_vx, control_signal_vy, control_signal_w]
        """

        control_signal_vx = 0.80975 * deviations[0] * 1.5 + 0.06663 * deviations[3] * 1.5
        control_signal_vy = 0.80971 * deviations[1] * 1.5 + 0.06163 * deviations[4] * 1.5
        control_signal_w = 0.8094 * deviations[2] * 1.5 + 0.03775 * deviations[5] * 1.5

        rotation_matrix = np.array([[np.cos(self.robot_state[2]), np.sin(self.robot_state[2])],
                                    [-np.sin(self.robot_state[2]), np.cos(self.robot_state[2])]])
        speed = np.array([[control_signal_vx],
                          [control_signal_vy]])
        rotated_speed = rotation_matrix @ speed
        return np.array([rotated_speed[0, 0], rotated_speed[1, 0], control_signal_w])

    def is_near_goal(self):
        dist = np.sqrt(
            (self.robot_state[0] - self.current_goal[0]) ** 2 + (self.robot_state[1] - self.current_goal[1]) ** 2)
        theta_err = abs(self.robot_state[2] - self.current_goal[2])
        return dist <= 0.1 and theta_err < np.pi / 20

    def callback_response(self, data):
        """Function parses data from STM into robot local frame speed data.
        Additionally it uses tfBuffer to get the pose of the robot

        Each time when data is published in topic "/secondary_robot/stm/response",
        this function is called.

        All processed data is saved in the parameter self.robot_state

        :param data: Data from the STM topic "/secondary_robot/stm/response"
        """
        splitted_data = data.data.split()

        if len(splitted_data) == 4:
            try:
                # Get current state of the robot
                trans = self.tfBuffer.lookup_transform("secondary_robot_odom", 'secondary_robot', rospy.Time())
                q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z,
                     trans.transform.rotation.w]
                angle = euler_from_quaternion(q)[2] % (2 * np.pi)
                self.angle_array.append(angle)
                unfolded_angle = unfold_angles(np.array(self.angle_array))[-1]

                # Get speed values form
                speed_vals = [float(token) for token in splitted_data]

                self.robot_state = np.array(
                    [trans.transform.translation.x, trans.transform.translation.y, unfolded_angle, speed_vals[1],
                     speed_vals[2], speed_vals[3]])
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Failed to get localization from PF")

    def tcallback_speed(self, event):
        """Function requests information about speed from the STM

        Each time the timer ticks with requested rate,
        this function publishes information with the request of the speed from the STM.

        """
        data = "0 9"  # Name of the command to request speed
        self.pub_stm_command.publish(data)  # Publish required command to the topic "/secondary_robot/stm/command"

    def calc_errors(self):
        """
        Uses the robot state and goal positions to calculate the deviation values

        :return error_arr: Array of deviations [x_deviation, y_deviation, theta_deviation,
                                                vx_deviation, vy_deviation, w_deviation]
        """
        error_arr = np.zeros(6)
        error_arr[0] = self.current_goal[0] - self.robot_state[0]
        error_arr[1] = self.current_goal[1] - self.robot_state[1]
        error_arr[2] = self.current_goal[2] - self.robot_state[2]
        error_arr[3] = -self.robot_state[3]
        error_arr[4] = -self.robot_state[4]
        error_arr[5] = -self.robot_state[5]
        return error_arr


def shutdown():
    """
    Function is called on shutdown.

    Sends 0 control to stm to stop the robot on shutdown.

    ***DOES NOT ALWAYS WORK***
    """
    control_string = "0" + " 0x08 " + "0 0 0"
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
