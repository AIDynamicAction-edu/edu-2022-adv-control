#!/usr/bin/env python3
import rospy
import warnings
import numpy as np
import do_mpc
import matplotlib
matplotlib.use('Agg')
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion
import tf2_ros


def wrap_angle(angle):
    """Function performs wrapping in the [0, 2pi] range.

    Args:
        angle (float): Given angle in [rad]

    Returns:
        wrapped_angle: Wrapped angle in [rad]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unfold_angles(angles):
    """Function performs unwrapping of the wrapped in range [0, 2pi] angle.

    Args:
        angles (np.array): Given angles in [rad]

    Returns:
        unwrapped_angles: Resulted unwrapped angles in [rad]
    """
    angles = wrap_angle(angles)
    delta = angles[1:] - angles[:-1]
    delta = np.where(delta > np.pi, delta - 2 * np.pi, delta)
    delta = np.where(delta < -np.pi, delta + 2 * np.pi, delta)
    return angles[0] + np.concatenate([np.zeros(1), np.cumsum(delta)], axis=0)


class MPC_Controller(object):
    def __init__(self):
        self.robot_state = np.zeros(6)
        self.angle_array = []  # Angle array to unfold angle
        self.flag = False  # End of path flag

        # Target global point to reach


        # Creating path vosmerka
        angle = np.linspace(np.pi, 3 * np.pi, 50)
        x = (0.5 * 2 ** 0.5 * np.cos(angle)) / (1 + np.sin(angle) ** 2) + 1.5
        y = (0.5 * 2 ** 0.5 * np.cos(angle) * np.sin(angle)) / (1 + np.sin(angle) ** 2)
        # self.path = np.array([[0.5, 0, 0],
        #                       [0.5, 0.15, np.pi/2],
        #                       [1.5, 0.15, np.pi],
        #                       [1.5, -0.15, 3 * np.pi / 2],
        #                       [0.5, -0.15, 2 * np.pi],
        #                       [0.5, 0, 2* np.pi],
        #                       [0, 0, 2*np.pi]])

        self.path = np.array([x, y, 10*angle]).T
        self.path = np.append(self.path, [[self.path[0, 0], self.path[0, 1], self.path[-1, 2]]], axis=0)
        self.path = np.append(self.path, [[0, 0, self.path[-1, 2]]], axis=0)

        self.current_goal = self.path[0]
        # self.current_speed_goal = self.speed_path[0]
        self.cnt = 0

        # do-mpc: Start ----------------------------------------------------
        RATE = 10  # Rate of the system [Hz]

        # Step 1. Creating model of the system
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)
        # State variables
        x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        y = model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
        theta = model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
        # Control variables
        Vx_set = model.set_variable(var_type='_u', var_name='Vx_set')
        Vy_set = model.set_variable(var_type='_u', var_name='Vy_set')
        W_set = model.set_variable(var_type='_u', var_name='W_set')
        # Three additional states for the true robot velocities
        Vx = model.set_variable(var_type='_x', var_name='Vx', shape=(1, 1))
        Vy = model.set_variable(var_type='_x', var_name='Vy', shape=(1, 1))
        W = model.set_variable(var_type='_x', var_name='W', shape=(1, 1))

        # path points
        x_ref = model.set_variable(var_type='_tvp', var_name='x_ref')
        y_ref = model.set_variable(var_type='_tvp', var_name='y_ref')
        theta_ref = model.set_variable(var_type='_tvp', var_name='theta_ref')

        # Set model parameters of the equations
        Tvx = model.set_variable('parameter', 'Tvx')
        Tvy = model.set_variable('parameter', 'Tvy')
        Tw = model.set_variable('parameter', 'Tw')


        # Right-hand side of the equations
        model.set_rhs('x', Vx)
        model.set_rhs('y', Vy)
        model.set_rhs('theta', W)
        model.set_rhs('Vx', 1 / Tvx * (Vx_set - Vx))
        model.set_rhs('Vy', 1 / Tvy * (Vy_set - Vy))
        model.set_rhs('W', 1 / Tw * (W_set - W))
        # Model setup
        model.setup()
        # Copy model as the field of the class instance
        self.model = model

        # Step 2. Configuring MPC Controller
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': 20,
            't_step': 1 / RATE,
            'n_robust': 1,
        }
        mpc.set_param(**setup_mpc)
        # Objective cost function
        lterm = (self.model.tvp['x_ref'] - self.model.x['x']) ** 2 + (
                        self.model.tvp['y_ref'] - self.model.x['y']) ** 2 + (
                                self.model.tvp['theta_ref'] - self.model.x['theta']) ** 2
        # lterm = (0.2 - self.model.x['Vx']) ** 2 + (0 - self.model.x['Vy']) ** 2 + (0 - self.model.x['W']) ** 2
        mterm = lterm
        mpc.set_objective(mterm=mterm, lterm=lterm)
        # Penalty for using actuators
        mpc.set_rterm(
            Vx_set=10,
            Vy_set=10,
            W_set=10,
        )
        # Setup bounds
        # Lower bounds on inputs
        mpc.bounds['lower', '_u', 'Vx_set'] = -0.8  # [m/s]
        mpc.bounds['lower', '_u', 'Vy_set'] = -0.8  # [m/s]
        mpc.bounds['lower', '_u', 'W_set'] = -2.5  # [m/s]
        # Upper bounds on inputs
        mpc.bounds['upper', '_u', 'Vx_set'] = 0.8  # [m/s]
        mpc.bounds['upper', '_u', 'Vy_set'] = 0.8  # [m/s]
        mpc.bounds['upper', '_u', 'W_set'] = 2.5  # [m/s]
        # Setup uncertain parameters

        dynamics_Vx = np.array([0.083*1.1])
        dynamics_Vy = np.array([0.077333*1.1])
        dynamics_W = np.array([0.044*1.1])
        mpc.set_uncertainty_values(
            Tvx=dynamics_Vx,
            Tvy=dynamics_Vy,
            Tw=dynamics_W
        )

        # set point to go to
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            for k in range(setup_mpc['n_horizon'] + 1):
                tvp_template['_tvp', k, 'x_ref'] = self.current_goal[0]
                tvp_template['_tvp', k, 'y_ref'] = self.current_goal[1]
                tvp_template['_tvp', k, 'theta_ref'] = self.current_goal[2]
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        # Controller setup
        mpc.setup()
        # Set initial position
        x0 = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)
        mpc.x0 = x0
        # Set initial guess
        mpc.set_initial_guess()

        # Copy mpc as the field of the class instance
        self.mpc = mpc

        # do-mpc: End ------------------------------------------------------
        #rospy.logwarn("end mpc")
        # ROS: Start -------------------------------------------------------
        # Tf listener for localization
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        # Creating ROS Publishers
        self.pub_stm_command = rospy.Publisher("/secondary_robot/stm/command", String,
                                               queue_size=1)  # Control command is published here
        self.pub_robot_command = rospy.Publisher("/secondary_robot/robot_command", String, queue_size=1)
        self.state_pub_header = rospy.Publisher("/secondary_robot/robot_state_header", Header, queue_size=1)

        # Creating ROS Subscribers
        rospy.Subscriber("/secondary_robot/stm/response", String, self.callback_response, queue_size=1)

        # Initialize poll timers
        self.timer = rospy.Timer(rospy.Duration(1. / RATE), self.tcallback_speed)
        self.timer_control = rospy.Timer(rospy.Duration(1. / RATE), self.tcallback_control, reset=True)
        # ROS: End ---------------------------------------------------------

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
        pose_dev = self.calc_errors()
        if not self.flag:
            if self.is_near_goal():
                # control_string = "0" + " 0x08 " + "0 0 0"
                # #rospy.logwarn("got to goal")
                # self.pub_stm_command.publish(control_string)

                if self.cnt < self.path.shape[0] - 1:
                    self.cnt += 1
                    self.current_goal = self.path[self.cnt]
                    # self.current_speed_goal = self.speed_path[self.cnt]
                    #rospy.logwarn(self.robot_state)
                else:
                    self.flag = True
            else:
                state = Header()
                state.stamp = rospy.Time.now()
                state.frame_id = "%s %s %s %s %s %s" % (
                    self.robot_state[0], self.robot_state[1], self.robot_state[2],
                    self.robot_state[3], self.robot_state[4], self.robot_state[5])
                rospy.logwarn(self.robot_state[3])
                self.state_pub_header.publish(state)
                control_signal = self.get_control_signal(self.robot_state[np.newaxis].T)

                # Publish data to STM
                control_string = "%f %f %f" % (control_signal[0], control_signal[1], control_signal[2])
                #rospy.logwarn("Goal %s" % self.current_goal)
                control_string = "0" + " 0x08 " + control_string
                self.pub_stm_command.publish(control_string)
                self.pub_robot_command.publish(control_string)

        else:
            control_string = "0" + " 0x08 " + "0 0 0"
            self.pub_stm_command.publish(control_string)
            self.flag = True

    def get_control_signal(self, x0):
        u0 = self.mpc.make_step(x0)

        rotation_matrix = np.array([[np.cos(self.robot_state[2]), np.sin(self.robot_state[2])],
                                    [-np.sin(self.robot_state[2]), np.cos(self.robot_state[2])]])
        speed = np.array([[u0[0, 0]],
                          [u0[1, 0]]])
        rotated_speed = rotation_matrix @ speed
        # rospy.logwarn("control %s" % rotated_speed[0,0])
        return np.array([rotated_speed[0, 0], rotated_speed[1, 0], u0[2, 0]])

    def is_near_goal(self):
        dist = np.sqrt(
            (self.robot_state[0] - self.current_goal[0]) ** 2 + (self.robot_state[1] - self.current_goal[1]) ** 2)
        theta_err = abs(self.robot_state[2] - self.current_goal[2])
        return dist <= 0.3 and theta_err < np.pi / 20

    def callback_response(self, data):
        """Function parses data from STM into robot local frame speed data.
        Each time when data is published in topic "/secondary_robot/stm/response",
        this function is called.
        :param data: Data from the STM topic "/secondary_robot/stm/response"
        :return: self.vx_local, self.vy_local, self.w_local
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
        # Calculate deviation between goal and current measurement
        error_arr = np.zeros(6)
        error_arr[0] = self.current_goal[0] - self.robot_state[0]
        error_arr[1] = self.current_goal[1] - self.robot_state[1]
        error_arr[2] = self.current_goal[2] - self.robot_state[2]
        error_arr[3] = -self.robot_state[3]
        error_arr[4] = -self.robot_state[4]
        error_arr[5] = -self.robot_state[5]
        return error_arr


def shutdown():
    control_string = "0" + " 0x08 " + "0 0 0"
    controller_node.pub_stm_command.publish(control_string)
    rospy.logwarn("MPC Controller is shutting down!")


if __name__ == '__main__':
    try:

        rospy.on_shutdown(shutdown)
        rospy.init_node('MPC_controller_node', anonymous=True)
        warnings.filterwarnings("error")
        #rospy.logwarn("!!!!!!!!!!!!!!!!!!!start mpc!!!!!!!!!!!!!!!!!!!!!!!!!")
        controller_node = MPC_Controller()
        #rospy.logwarn("Controller created")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
