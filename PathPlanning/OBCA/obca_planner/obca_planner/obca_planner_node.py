import rclpy
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_auto_perception_msgs.msg import PredictedObjects
from autoware_auto_vehicle_msgs.msg import SteeringReport
from autoware_auto_mapping_msgs.msg import HADMapBin
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, AccelWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
from obca_planner.obca import *
from obca_planner.unit import *
from visualization_msgs.msg import Marker
import functools


def IsDiffPose(p1: Pose, p2: Pose, dist_err, ang_err):
    dist_square = (p1.position.x - p2.position.x) ** 2 + (p1.position.y - p2.position.y) ** 2
    quat1 = [p1.orientation.x, p1.orientation.y, p1.orientation.z, p1.orientation.w]
    quat2 = [p2.orientation.x, p2.orientation.y, p2.orientation.z, p2.orientation.w]
    angle1 = ConvertQuaternionToYaw(quat1)
    angle2 = ConvertQuaternionToYaw(quat2)
    angle_error = AngleDiff(angle1, angle2)
    if dist_square > dist_err or math.fabs(angle_error) > ang_err:
        return True
    return False


def OnMapCallBack(msg):
    global raw_data
    raw_data['map'] = msg


def OnTrajecotryCallBack(msg: Trajectory):
    global raw_data, has_new_traj
    if len(msg.points) <= 1 or raw_data['goal'] is None: return
    if has_new_traj: raw_data['traj'] = msg
    if raw_data['traj'] is None or len(raw_data['traj'].points) != len(msg.points) or IsDiffPose(
            raw_data['traj'].points[-1].pose,
            msg.points[-1].pose, 1e-3, 1e-3):
        has_new_traj = True
        return


def HasStop(odom_buffer):
    while len(odom_buffer) > 0:
        time_diff = node.get_clock().now().seconds_nanoseconds()[0] - odom_buffer[0].header.stamp.sec
        if time_diff < 3.0:
            break
        odom_buffer.pop(0)

    for i in range(len(odom_buffer)):
        if math.fabs(odom_buffer[i].twist.twist.linear.x) > 1e-3:
            return False
    return True


def OnOdometryCallBack(msg):
    global raw_data, odom_buffer
    raw_data['odom'] = msg
    odom_buffer.append(raw_data['odom'])


def OnGoalCallBack(msg):
    global raw_data
    raw_data['goal'] = msg


def OnObstacleCallBack(msg):
    global raw_data
    raw_data['obs'] = msg


def OnAcclerationCallBack(msg):
    global raw_data
    raw_data['acc'] = msg


def OnSteeringCallBack(msg):
    global raw_data
    raw_data['steer'] = msg


def ConvertDataFormat(raw_data):
    robot = raw_data['odom'].pose
    quat = [robot.pose.orientation.x, robot.pose.orientation.y, robot.pose.orientation.z, robot.pose.orientation.w]
    x0 = np.zeros((4, 1))
    x0[0, 0] = robot.pose.position.x
    x0[1, 0] = robot.pose.position.y
    x0[2, 0] = ConvertQuaternionToYaw(quat)
    x0[3, 0] = raw_data['odom'].twist.twist.linear.x

    goal = raw_data['goal'].pose
    quat = [goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w]
    xF = np.zeros((4, 1))
    xF[0, 0] = goal.position.x
    xF[1, 0] = goal.position.y
    xF[2, 0] = ConvertQuaternionToYaw(quat)
    xF[3, 0] = 0

    u0 = np.array([[raw_data['acc'].accel.accel.linear.x], [raw_data['steer'].steering_tire_angle]])

    N = len(raw_data['traj'].points)
    ref_traj = np.zeros((4, N))
    for i in range(N):
        traj_pnt = raw_data['traj'].points[i]
        quat = [traj_pnt.pose.orientation.x, traj_pnt.pose.orientation.y, traj_pnt.pose.orientation.z,
                traj_pnt.pose.orientation.w]
        ref_traj[0, i] = traj_pnt.pose.position.x
        ref_traj[1, i] = traj_pnt.pose.position.y
        ref_traj[2, i] = ConvertQuaternionToYaw(quat)
        ref_traj[3, i] = traj_pnt.longitudinal_velocity_mps

    obstacles = []
    if raw_data['obs'] is not None:
        for obj in raw_data['obs'].objects:
            pose = obj.kinematics.initial_pose_with_covariance.pose
            posi = [pose.pose.position.x, pose.pose.position.y]
            quat = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                    pose.pose.orientation.w]
            centroid = [posi[0], posi[1], ConvertQuaternionToYaw(quat)]
            shape = [obj.shape.dimensions.x, obj.shape.dimensions.y]
            obs = ExtractRectangularContourPoints(centroid, shape)
            obstacles.append(obs)

    XYbounds = [-np.inf, np.inf, -np.inf, np.inf]
    if raw_data['map'] is not None:
        pass
    return x0, xF, u0, ref_traj, obstacles, XYbounds


def IsDataReady(raw_data):
    if raw_data['odom'] is None or raw_data['goal'] is None or raw_data['traj'] is None or raw_data['map'] is None:
        return False
    return True


def GetIndexInterval(traj):
    idx = 0
    index_interval = []
    vel_list = traj[3, :]
    orientation = np.sign(vel_list[1])
    for i in range(1, len(vel_list) - 1):
        if (vel_list[i - 1] * vel_list[i] < 0):
            orientation = -1 * orientation
            index_interval.append((idx, i))
            idx = i
    index_interval.append((idx, len(vel_list) - 1))
    return index_interval

def UpdateTargetIndex(traj, seg_idx_list):
    start_index, end_index = seg_idx_list[0]
    if HasStop(odom_buffer):
        p1 = [raw_data['odom'].pose.pose.position.x, raw_data['odom'].pose.pose.position.y]
        p2 = [traj[0, end_index], traj[1, end_index]]
        if CalcDistance(p1, p2) < 1.0:
            seg_idx_list.pop(0)
    has_finish = True if seg_idx_list == [] else False
    return start_index, end_index, has_finish


def plot_arrow(traj, topic):
    posearray = PoseArray()
    posearray.header.frame_id = 'map'
    posearray.header.stamp = node.get_clock().now().to_msg()
    for i in range(len(traj.points)):
        p = traj.points[i].pose
        posearray.poses.append(p)
    node.create_publisher(PoseArray, topic, 1).publish(posearray)


def plot_line(traj, topic, color=[1.0, 0.0, 0.0]):
    # 绘制轨迹点
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.1
    marker.color.a = 1.0
    marker.color.g = color[0]
    marker.color.b = color[1]
    marker.color.r = color[2]

    traj_points = []
    for i in range(len(traj.points)):
        point = Point()
        point.x = traj.points[i].pose.position.x
        point.y = traj.points[i].pose.position.y
        point.z = 0.0
        traj_points.append(point)
    marker.points = traj_points
    node.create_publisher(Marker, topic, 1).publish(marker)


def GetSegmentTrajectory(trajectory, start_index, end_index):
    seg_traj = Trajectory()
    seg_traj.header.stamp = node.get_clock().now().to_msg()
    seg_traj.header.frame_id = 'map'
    for i in range(start_index, end_index):
        tp = TrajectoryPoint()
        tp.pose.position.x = trajectory[0, i]
        tp.pose.position.y = trajectory[1, i]
        tp.pose.position.z = 0.0
        quat = ConvertYawToQuaternion(trajectory[2, i])
        tp.pose.orientation.x = quat[0]
        tp.pose.orientation.y = quat[1]
        tp.pose.orientation.z = quat[2]
        tp.pose.orientation.w = quat[3]
        tp.longitudinal_velocity_mps = trajectory[3, i]
        seg_traj.points.append(tp)
    return seg_traj


ego = None
traj_pub = None
has_new_traj = False
trajectory = None
seg_idx_list = []
odom_buffer = []
raw_data = {
    "traj": None,
    "odom": None,
    "goal": None,
    "obs": None,
    "acc": None,
    "steer": None,
    "map": None
}


def OnTimer(node):
    global ego, traj_pub, raw_data, has_new_traj, trajectory, seg_idx_list, ref_traj, ref_input
    if ego is None:
        ego = GetVehicleInfo(node)
    if traj_pub is None:
        traj_pub = node.create_publisher(Trajectory, '/planning/scenario_planning/parking/trajectory',
                                         rclpy.qos.qos_profile_system_default)
    if not IsDataReady(raw_data):
        return
    if has_new_traj:
        x0, xF, u0, ref_traj, obstacles, XYbounds = ConvertDataFormat(raw_data)
        ref_input = CalculateReferenceInput(ref_traj, ego)
        trajectory, _, _ = planning(x0, xF, u0, ego, XYbounds, obstacles, ref_traj, ref_input, 0.02)
        seg_idx_list = GetIndexInterval(trajectory)
        has_new_traj = False
    if trajectory is not None and len(seg_idx_list) != 0:
        start_index, end_index, has_finish = UpdateTargetIndex(trajectory, seg_idx_list)
        if not has_finish:
            seg_traj = GetSegmentTrajectory(trajectory, start_index, end_index)
            traj_pub.publish(seg_traj)

        # debug
        all_traj = GetSegmentTrajectory(trajectory, 0, trajectory.shape[1])
        ref_traj1 = GetSegmentTrajectory(ref_traj, 0, ref_traj.shape[1])
        plot_arrow(all_traj, "/obca_all_traj_ori")
        plot_arrow(seg_traj, "/obca_seg_traj_ori")
        plot_arrow(ref_traj1, "/obca_ref_traj_ori")
        plot_line(all_traj, "/obca_all_traj")
        plot_line(seg_traj, "/obca_seg_traj")
        plot_line(ref_traj1, "/obca_ref_traj", [0.0, 1.0, 0.0])

        temp_yaw = []
        temp_yaw_car = []
        temp_yaw_ref_input2 = []
        ref_traj2 = np.zeros(ref_traj.shape)
        for i in range(1, ref_input.shape[1]):
            ref_traj2[0, i] = ref_traj[0, i]
            ref_traj2[1, i] = ref_traj[1, i]
            ref_traj2[2, i] = ref_input[1, i - 1] + ref_traj[2, i]
            temp_yaw.append(np.rad2deg(ref_input[1, i - 1]))
            temp_yaw_car.append(np.rad2deg(ref_traj[2, i]))
            temp_yaw_ref_input2.append(np.rad2deg(ref_input[1, i - 1] + ref_traj[2, i]))
        ref_traj21 = GetSegmentTrajectory(ref_traj2, 0, ref_traj2.shape[1])
        plot_arrow(ref_traj21, "/obca_ref_input_ori")


def GetVehicleInfo(node):
    max_vel = node.declare_parameter('vel_lim', 50.0).value  # robobus 10.0 sample 50.0
    min_vel = 0.25  # motion_velocity_smoother 起步速度至少0.25
    max_steer = node.declare_parameter('steer_lim', 0.7).value  # robobus 1.0 sample 1.0
    max_acc = node.declare_parameter('vel_rate_lim', 10.).value  # robobus 2.0 sample 0.1
    max_steer_rate = node.declare_parameter('steer_rate_lim', 0.7).value  # robobus 5.0 sample 5.0, mpc 0.7
    wheel_base = node.declare_parameter('wheel_base', 2.79).value  # robobus 3.020 sample 2.79
    front_overhang = node.declare_parameter('front_overhang', 1.0).value  # robobus 0.400 sample 1.0
    rear_overhang = node.declare_parameter('rear_overhang', 1.1).value  # robobus 0.400 sample 1.1
    vehicle_length = wheel_base + front_overhang + rear_overhang
    wheel_tread = node.declare_parameter('wheel_tread', 1.64).value  # robobus 1.618 sample 1.64
    left_overhang = node.declare_parameter('left_overhang', 0.128).value  # robobus 0.0 sample 0.128
    right_overhang = node.declare_parameter('right_overhang', 0.128).value  # robobus 0.0 sample 0.128
    vehicle_width = wheel_tread + left_overhang + right_overhang
    rear2center = vehicle_length / 2.0 - rear_overhang  # rear2center = 0

    vehicle_info = {
        "max_vel": max_vel,
        'min_vel': min_vel,
        "max_steer": max_steer,
        "max_acc": max_acc,
        "max_steer_rate": max_steer_rate,
        "wheel_base": wheel_base,
        "width": vehicle_width,
        "length": vehicle_length,
        "rear_overhang": rear_overhang,
        "rear2center": rear2center
    }
    node.get_logger().info("xxxxx wheel_base is {:.3f}".format(wheel_base))
    node.get_logger().info("xxxxx front_overhang is {:.3f}".format(front_overhang))
    node.get_logger().info("xxxxx rear_overhang is {:.3f}".format(rear_overhang))
    node.get_logger().info("xxxxx wheel_tread is {:.3f}".format(wheel_tread))
    node.get_logger().info("xxxxx left_overhang is {:.3f}".format(left_overhang))
    node.get_logger().info("xxxxx right_overhang is {:.3f}".format(right_overhang))
    return vehicle_info


if __name__ == '__main__':
    rclpy.init()

    node = rclpy.create_node('obca_planner_node')
    # 通过obca_planner.launch.py 启动时使用
    # map_sub = node.create_subscription(HADMapBin, '~/input/map', OnMapCallBack, QoSProfile(
    #     reliability=ReliabilityPolicy.RELIABLE,
    #     history=HistoryPolicy.KEEP_LAST,
    #     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    #     depth=1
    # ))
    # traj_sub = node.create_subscription(Trajectory, '~/input/trajectory', OnTrajecotryCallBack, 10)
    # obs_sub = node.create_subscription(PredictedObjects, '~/input/obstacles', OnObstacleCallBack, 10)
    # odom_sub = node.create_subscription(Odometry, '~/input/odometry', OnOdometryCallBack, 10)
    # goal_pose_sub = node.create_subscription(PoseStamped, '~/input/goal_pose', OnGoalCallBack, 10)
    # acc_sub = node.create_subscription(AccelWithCovarianceStamped, '~/input/acceleration', OnAcclerationCallBack, 10)
    # steer_sub = node.create_subscription(SteeringReport, '~/input/steering', OnSteeringCallBack, 10)
    # traj_pub = node.create_publisher(Trajectory, '~/output/trajectory', 1)

    # 单独启动节点
    map_sub = node.create_subscription(HADMapBin, '/map/vector_map', OnMapCallBack,
                                       QoSProfile(
                                           reliability=ReliabilityPolicy.RELIABLE,
                                           history=HistoryPolicy.KEEP_LAST,
                                           durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                                           depth=1
                                       ))

    traj_sub = node.create_subscription(Trajectory, '/planning/scenario_planning/parking/hyridAStar/trajectory',
                                        OnTrajecotryCallBack,
                                        rclpy.qos.qos_profile_system_default)
    obs_sub = node.create_subscription(PredictedObjects, '/perception/object_recognition/objects',
                                       OnObstacleCallBack,
                                       10)
    odom_sub = node.create_subscription(Odometry, '/localization/kinematic_state',
                                        OnOdometryCallBack, 10)
    goal_pose_sub = node.create_subscription(PoseStamped, '/planning/mission_planning/goal',
                                             OnGoalCallBack, 10)
    acc_sub = node.create_subscription(AccelWithCovarianceStamped, '/localization/acceleration',
                                       OnAcclerationCallBack,
                                       10)
    steer_sub = node.create_subscription(SteeringReport, '/vehicle/status/steering_status',
                                         OnSteeringCallBack, 10)
    timer = node.create_timer(timer_period_sec=0.1, callback=functools.partial(OnTimer, node))
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
