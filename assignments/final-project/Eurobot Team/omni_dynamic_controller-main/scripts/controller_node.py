#!/usr/bin/env python3
import warnings
import numpy as np
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
import tf2_ros

# rospy.logwarn("Hello world!")
# Define control rate
RATE_get_local_robot_speed = 50
RATE_publish_control = 50


def cvt_global2local(global_point, src_point):
    size = global_point.shape[-1]
    x1, y1, a1 = 0, 0, 0
    if size == 3:
        x1, y1, a1 = global_point.T
    elif size == 2:
        x1, y1 = global_point.T
    X, Y, A = src_point.T
    x = x1 * np.cos(A) + y1 * np.sin(A) - X * np.cos(A) - Y * np.sin(A)
    y = -x1 * np.sin(A) + y1 * np.cos(A) + X * np.sin(A) - Y * np.cos(A)
    a = (a1 - A + np.pi) % (2 * np.pi) - np.pi
    if size == 3:
        return np.array([x, y, a]).T
    elif size == 2:
        return np.array([x, y]).T
    else:
        return

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unfold_angles(angles):
    angles = wrap_angle(angles)
    delta = angles[1:] - angles[:-1]
    delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)
    delta = np.where(delta < -np.pi, delta + 2 * np.pi, delta)
    return angles[0] + np.concatenate([np.zeros(1), np.cumsum(delta)], axis=0)


class ControllerNode(object):
    """
    Controller operates in the local coordinate system assigned with the robot.
    Localization coordinates should be transformed into local robot coordinates.
    Target point should be transformed into local coordinate system.
    """

    def __init__(self):
        self.robot_pose = None
        self.Int_Err = None
        self.start = rospy.get_time()
        self.flag = False

        # tf listener for localization

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # Creating trajectory
        # keypoints for the trajectory
        # poses = np.array([[1, 0, 0],
        #                  [1, 0.5, np.pi / 2],
        #                  [0, 0, np.pi]])
        # creating trajectory [[x], [y], [theta]]
        # x = np.array([0])
        # y = np.array([0])
        # theta = np.array([0])
        # num_of_points = 100
        # for pose in poses:
        #     x_lin = np.linspace(x[-1], pose[0], num_of_points)
        #     y_lin = np.linspace(y[-1], pose[1], num_of_points)
        #     theta_lin = np.linspace(theta[-1], pose[2], num_of_points)
        #
        #     x = np.concatenate([x, x_lin])
        #     y = np.concatenate([y, y_lin])
        #     theta = np.concatenate([theta, theta_lin])
        #
        # self.trajectory = np.vstack([x, y, theta])

        # creating delta array [[dx, dy, dtheta]]
        # delta = []
        # for i in range(self.trajectory.shape[1] - 1):
        #     delta.append(np.array(
        #         [self.trajectory[0, i + 1] - self.trajectory[0, i], self.trajectory[1, i + 1] - self.trajectory[1, i],
        #          self.trajectory[2, i + 1] - self.trajectory[2, i]]))
        # self.deltas = np.array(delta)

        # Current goal

        # deviation
        self.x_dev_err = 0
        self.y_dev_err = 0
        self.theta_dev_err = 0
        self.vx_dev_err = 0
        self.vy_dev_err = 0
        self.w_dev_err = 0

        # Global system
        self.x_target_global = 1
        self.y_target_global = 0
        self.theta_target_global = 0

        # Local system
        angle = np.linspace(np.pi, 3*np.pi, 150)
        x = (0.5 * 2 ** 0.5 * np.cos(angle)) / (1 + np.sin(angle) ** 2) + 1.5
        y = (0.5 * 2 ** 0.5 * np.cos(angle) * np.sin(angle)) / (1 + np.sin(angle) ** 2)
        self.goals = np.array([x, y, 10*angle]).T
        self.goals = np.append(self.goals, [[self.goals[0, 0], self.goals[0, 1], self.goals[-1, 2]]], axis=0)
        self.goals = np.append(self.goals, [[0, 0, self.goals[-1, 2]]], axis=0)
        self.x_goal_local = self.goals[0, 0]
        self.y_goal_local = self.goals[0, 1]
        self.theta_goal_local = self.goals[0, 2]

        self.cnt = 0

        # Integrals initialization
        self.x_err_int = 0  # Возможно нужно перевести все в локальную систему координат либо изменить начальные условия на текущую координату
        self.y_err_int = 0
        self.theta_err_int = 0

        # Setting initial speed outputs
        self.vx_local = 0
        self.vy_local = 0
        self.w_local = 0

        # Initial poses
        self.x_local = 0  # Position X in the local coordinate system
        self.y_local = 0  # Position Y in the local coordinate system
        self.theta_local = 0  # Position Theta in the local coordinate system

        self.x_global_init = None
        self.y_global_init = None
        self.w_global_init = None

        # Initialize augmented LQR
        self.K_fsfb = np.genfromtxt('/home/rpi2/catkin_ws/src/reset_22/omni_dynamic_controller/scripts/K_fsfb.csv',
                                    delimiter=',')
        self.K_int = np.genfromtxt('/home/rpi2/catkin_ws/src/reset_22/omni_dynamic_controller/scripts/K_int.csv',
                                   delimiter=',')

        # Creating ROS Publishers
        self.pub_stm_command = rospy.Publisher("/secondary_robot/stm/command", String, queue_size=1)
        self.pub_robot_command = rospy.Publisher("/secondary_robot/robot_command", String, queue_size=1)
        self.pub_int_error = rospy.Publisher("/secondary_robot/int_error", String, queue_size=1)
        self.state_pub = rospy.Publisher("/secondary_robot/robot_state", String, queue_size=1)
        self.state_pub_header = rospy.Publisher("/secondary_robot/robot_state_header", Header, queue_size=1)

        # Creating ROS Subscribers
        rospy.Subscriber("/secondary_robot/stm/response", String, self.callback_response, queue_size=1)
        rospy.Subscriber("/secondary_robot/filtered_coords", PoseWithCovarianceStamped, self.callback_localization,
                         queue_size=1)
        #rospy.Subscriber("/secondary_robot/odom", Odometry, self.callback_odometry, queue_size=1)

        # Initialize poll timers
        self.timer = rospy.Timer(rospy.Duration(1. / RATE_get_local_robot_speed), self.tcallback_speed)
        self.timer_control = rospy.Timer(rospy.Duration(1. / RATE_publish_control), self.tcallback_control, reset=True)

        # needed to unfold angles
        self.angle_array = []

    def tcallback_control(self, event):
        """Function publishes control current signal

        Each time the timer ticks with requested rate,
        this function:
            1) Calculates state of the robot self.X: [x_local, y_local, theta_local, vx_local, vy_local, w_local]
            2) It calls function to publish command for the STM to get the robot speed
            3) It initializes Int_Err
            4) It publishes robot state X
            5) Calculates Control from Int part and FSFB part and gets summary control vector
            6) Publishes control command for the STM
        """

        try:
            trans = self.tfBuffer.lookup_transform("secondary_robot_odom", 'secondary_robot', rospy.Time())
            self.x_local = trans.transform.translation.x
            self.y_local = -trans.transform.translation.y
            q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z,
                 trans.transform.rotation.w]

            # Get global angle
            angle = euler_from_quaternion(q)[2] % (2 * np.pi)
            self.angle_array.append(angle)
            unfolded_angle = unfold_angles(np.array(self.angle_array))[-1]
            self.theta_local = unfolded_angle
            # rospy.logwarn("%s %s" % (self.x_local, self.theta_local))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to get localization from PF")
            return 0

        self.integral_errors()
        self.start = rospy.get_time()
        # Defines robot state at each time step
        # if self.Int_Err is not None and not self.flag:
        if not self.flag:

            self.X = \
            np.array([self.x_local, self.y_local, self.theta_local, self.vx_local, self.vy_local, self.w_local])[
                np.newaxis].T

            dist = np.sqrt((self.x_local - self.x_goal_local) ** 2 + (self.y_local - self.y_goal_local) ** 2)
            #
            if dist <= 0.1 and abs(self.theta_goal_local - self.theta_local) < np.pi / 2:
                control_string = "0" + " 0x08 " + "0 0 0"
                rospy.logwarn("control cmd: %s" % control_string)
                self.pub_stm_command.publish(control_string)
                self.Int_Err = np.array([[0],
                                         [0],
                                         [0]])

                if self.cnt <= self.goals.shape[0]:

                    self.x_goal_local = self.goals[self.cnt, 0]
                    self.y_goal_local = self.goals[self.cnt, 1]
                    self.theta_goal_local = self.goals[self.cnt, 2]
                    self.cnt += 1
                else:
                    self.flag = True
            else:
                #rospy.logwarn("state X: %s %s %s %s %s %s" % (
                #self.X.T[0, 0], self.X.T[0, 1], self.X.T[0, 2], self.X.T[0, 3], self.X.T[0, 4], self.X.T[0, 5]))
                state = Header()
                state.stamp = rospy.Time.now()
                state.frame_id = "%s %s %s %s %s %s" % (
                self.X.T[0, 0], self.X.T[0, 1], self.X.T[0, 2], self.X.T[0, 3], self.X.T[0, 4], self.X.T[0, 5])
                self.state_pub_header.publish(state)
                # Calculate control vector
                # rospy.logwarn('K_int %s' % self.K_int)
                # Control_Int_Part = self.K_int @ self.Int_Err
                # Control_FSFB_Part = self.K_fsfb @ self.X
                # Control_Vector = Control_Int_Part - Control_FSFB_Part

                # rospy.logwarn("vx %s" % self.vx_dev_err)
                # rospy.logwarn("x %s" % self.x_dev_err)
                #
                # rospy.logwarn("vy %s" % self.vy_dev_err)
                # rospy.logwarn("y %s" % self.y_dev_err)
                #
                # rospy.logwarn("w %s" % self.w_dev_err)
                # rospy.logwarn("theta %s" % self.theta_dev_err)



                control_signal_vx = 0.80975 * self.x_dev_err * 2.5 - 0.06663 * self.vx_dev_err * 1.5
                control_signal_vy = -(0.80971 * self.y_dev_err * 2.5 + 0.06163 * self.vy_dev_err * 1.5)
                control_signal_w = 0.8094 * self.theta_dev_err * 2.5 - 0.03775 * self.w_dev_err * 1.5


                speed = np.array([[np.cos(self.theta_local), np.sin(self.theta_local)],
                                  [-np.sin(self.theta_local), np.cos(self.theta_local)]]) @ np.array([[control_signal_vx], [control_signal_vy]])

                rospy.logwarn("goal %s %s %s" % (self.x_goal_local, self.y_goal_local, self.theta_goal_local) )
                control_signal_vx = speed[0, 0]
                control_signal_vy = speed[1, 0]
                # for i in range(3):
                #     if abs(Control_Vector[i]) > 1:
                #         Control_Vector[i] = np.sign(Control_Vector[i]) * 1

                # Publish data to STM
                control_string = "%f %f %f" % (
                    control_signal_vx, control_signal_vy, control_signal_w)
                # rospy.logwarn("control cmd: %s" % control_string)

                control_string = "0" + " 0x08 " + control_string
                self.pub_stm_command.publish(control_string)
                self.pub_robot_command.publish(control_string)
                # int_error_string = "%f %f %f" % (self.Int_Err.T[0, 0], self.Int_Err.T[0, 1], self.Int_Err.T[0, 2])
                # self.pub_int_error.publish(int_error_string)

        else:
            control_string = "0" + " 0x08 " + "0 0 0"
            #rospy.logwarn("control cmd: %s" % control_string)
            self.pub_stm_command.publish(control_string)

            self.Int_Err = np.array([[0],
                                     [0],
                                     [0]])
            self.flag = True

    def callback_response(self, data):
        """Function parses data from STM into robot local frame speed data.

        Each time when data is published in topic "/secondary_robot/stm/response",
        this function is called.

        :param data: Data from the STM topic "/secondary_robot/stm/response"
        :return: self.vx_local, self.vy_local, self.w_local
        """
        # rospy.logwarn("speed data: %s" % data.data)
        splitted_data = data.data.split()
        # Subscribe to STM data
        # Kinematics Data Parse
        # rospy.logwarn(splitted_data)
        if len(splitted_data) == 4:

            speed_vals = [float(token) for token in splitted_data]
            self.vx_local = speed_vals[1]  # Forward Speed vx_local
            self.vy_local = speed_vals[2]  # Lateral Speed vy_local
            self.w_local = speed_vals[3]  # Angular Speed w_local
            # delta_time = rospy.get_time() - self.start
            # Wheel odometry localization
            # self.x_local += self.vx_local * delta_time #1 / RATE_get_local_robot_speed
            # self.y_local += self.vy_local * delta_time #1 / RATE_get_local_robot_speed
            # self.theta_local += 0#wrap_angle(self.w_local * delta_time) #* 1 / RATE_get_local_robot_speed)

        else:
            pass
            # rospy.logwarn("not speed value")

    def tcallback_speed(self, event):
        """Function requests information about speed from the STM

        Each time the timer ticks with requested rate,
        this function publishes information with the request of the speed from the STM.

        :return: Publish command into the topic "/secondary_robot/stm/command"
        """
        data = "0 9"  # Name of the command to request speed
        self.pub_stm_command.publish(data)  # Publish required command to the topic "/secondary_robot/stm/command"

    def callback_localization(self, data):
        """Function return global localization coordinates of the robot

        Each time information is published in topic "/secondary_robot/filtered_coords",
        this function is called.

        :param data:
        :return:
        """
        # Get global quaternion
        q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z,
             data.pose.pose.orientation.w]

        # Get global angle
        # angle = wrap_angle(euler_from_quaternion(q)[2] % (2 * np.pi))

        # self.robot_pose = np.array(
        #     [data.header.stamp.to_sec(), data.pose.pose.position.x, data.pose.pose.position.y, wrap_angle(angle)])
        #
        # if (self.x_global_init is None):
        #     self.x_global_init = self.robot_pose[1]
        #     self.y_global_init = self.robot_pose[2]
        #     self.w_global_init = self.robot_pose[3]

        # rospy.logwarn("robot pose: %s" % self.robot_pose)

        # Localization Data Parse
        # self.x_local = self.robot_pose[1] - self.x_global_init # Position X
        # self.y_local = self.robot_pose[2] - self.y_global_init  # Position Y
        # self.theta_local = wrap_angle(self.robot_pose[3] - self.w_global_init)  # Angle Theta

        # self.integral_errors()  # call integral_errors after receiving new message

    def integral_errors(self):
        # Calculate deviation between goal and current measurement
        self.x_dev_err = self.x_goal_local - self.x_local
        self.y_dev_err = self.y_goal_local - self.y_local
        self.theta_dev_err = self.theta_goal_local - self.theta_local
        self.vx_dev_err = -self.vx_local
        self.vy_dev_err = -self.vy_local
        self.w_dev_err = -self.w_local
        # Integrals Data
        # self.x_err_int += x_dev_err  # Intergal of Position Error X
        # self.y_err_int += y_dev_err  # Intergal of Position Error Y
        # self.theta_err_int += theta_dev_err  # Intergal of Angle Error Theta
        # Assign column vector of integral errors  
        # self.Int_Err = np.array([[self.x_err_int],
        #                          [self.y_err_int],
        #                          [self.theta_err_int]])

    def callback_odometry(self, data):
        rospy.logwarn("odometry!")
        self.x_local = data.pose.pose.position.x
        self.y_local = data.pose.pose.position.y
        q = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z,
             data.pose.pose.orientation.w]

        # Get global angle
        self.theta_local = 0  # euler_from_quaternion(q)[2] % (2 * np.pi)


def shutdown():
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
