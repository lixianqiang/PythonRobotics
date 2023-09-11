import sys
import casadi as ca
import numpy as np
import rclpy
import math
import copy

from rclpy.node import Node
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_auto_perception_msgs.msg import PredictedObjects
from autoware_auto_vehicle_msgs.msg import SteeringReport
from autoware_auto_mapping_msgs.msg import HADMapBin
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from geometry_msgs.msg import AccelWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
from obca_planner.obca import *
from obca_planner.unit import *

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# def SegmentPath(ref_path):
#     x_list, y_list = ref_path[0], ref_path[1]
#     shift_index_list = []
#     for i in range(1, len(x_list) - 1):
#         p1 = [x_list[i - 1], y_list[i - 1]]
#         p2 = [x_list[i], y_list[i]]
#         p3 = [x_list[i + 1], y_list[i + 1]]
#         if (IsShiftPoint(p2, p1, p3)):
#             shift_index_list.append(i)
#     seg_path_list = []
#     for i in range(len(shift_index_list) - 1):
#         start_index = shift_index_list[i]
#         end_index = shift_index_list[i + 1]
#         seg_path = ref_path[:, start_index:end_index]
#         seg_path_list.append(seg_path)
#     return seg_path_list


# def SamplePathByDistance(ref_path, sample_distance):
#     for seg_path in all_path:
#         x = seg_path[0]
#         y = seg_path[1]
#         spline = Spline2D(x, y)
#         s = np.arange(0, spline.s[-1], sample_distance)
#         if spline.s[-1] - spline.s[-1] <= 0.1:
#             s = s[:-1]
#         for i in range(len(s)):
#             ix, iy = spline.calc_position(s[i])
#             iyaw = spline.calc_yaw(s[i])
#             rx.append(ix)
#             ry.append(iy)
#             ryaw.append(iyaw)
#     return rx, ry, ryaw

raw_traj = None
has_new_traj = False
raw_odom = None
odom_buffer = []
raw_goal = None
raw_steer = None
ego = None
once = None
all_traj = None
node = None
traj_pub = None
raw_obs = None
raw_acc = None
seg_idx_list = []
trajectory = None
has_new_goal = None


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


def IsDiffTraj(traj1: Trajectory, traj2: Trajectory):
    if len(traj1.points) != len(traj2.points):
        return True
    for p1, p2 in zip(traj1.points, traj2.points):
        if IsDiffPose(p1.pose, p2.pose, 0.25, np.pi / 18):
            return True
    return False


raw_map = None


def OnMapCallBack(msg):
    global raw_map
    raw_map = msg


def OnTrajecotryCallBack(msg: Trajectory):
    global raw_traj, has_new_traj, has_new_goal
    if has_new_goal and len(msg.points) > 1:
        # p1 = [raw_goal.pose.position.x, raw_goal.pose.position.y]
        # p2 = [msg.points[-1].pose.position.x, msg.points[-1].pose.position.y]
        if not IsDiffPose(raw_goal.pose, msg.points[-1].pose, 2.0, np.pi / 18):
            raw_traj = msg
            has_new_traj = True
            has_new_goal = False


def HasStop(odom_buffer):
    for i in range(len(odom_buffer)):
        if math.fabs(odom_buffer[i].twist.twist.linear.x) > 1e-3:
            return False
    return True


def OnOdometryCallBack(msg):
    global raw_odom, odom_buffer, node
    raw_odom = msg
    odom_buffer.append(raw_odom)
    while len(odom_buffer) > 0:
        time_diff = node.get_clock().now().seconds_nanoseconds()[0] - odom_buffer[0].header.stamp.sec
        if time_diff < 3.0:
            break
        odom_buffer.pop(0)


def OnGoalCallBack(msg):
    global raw_goal, has_new_goal
    if raw_goal is None or IsDiffPose(raw_goal.pose, msg.pose, 0.1, np.pi / 18):
        raw_goal = msg
        has_new_goal = True


def OnObstacleCallBack(msg):
    global raw_obs
    raw_obs = msg


def OnAcclerationCallBack(msg):
    global raw_acc
    raw_acc = msg


def OnSteeringCallBack(msg):
    global raw_steer
    raw_steer = msg


def CalculateReferenceInput(ref_path):
    steer_list = []
    x_list, y_list, vel_list = ref_path[0], ref_path[1], ref_path[3]
    orientation = np.sign(vel_list[0])
    for i in range(1, len(x_list) - 1):
        if (vel_list[i - 1] * vel_list[i] < 0):
            orientation = -1 * orientation
            steer_list.append(steer_list[-1])
        else:
            p1 = [x_list[i - 1], y_list[i - 1]]
            p2 = [x_list[i], y_list[i]]
            p3 = [x_list[i + 1], y_list[i + 1]]
            k = CalcCurvature(p2, p1, p3)
            steer_list.append(orientation * math.atan2(0.5 * ego["wheel_base"] * k, 1))
    steer_list.append(steer_list[-1])
    acc_list = [0 for i in range(len(steer_list))]
    ref_input = np.vstack((acc_list, steer_list))
    return ref_input


def ConvertDataFormat(data_type, data):
    global node
    if data_type == "trajectory":
        N = len(data.points)
        traj = np.zeros((4, N))
        for i in range(N):
            traj_point = data.points[i].pose
            quaternion = [traj_point.orientation.x, traj_point.orientation.y, traj_point.orientation.z,
                          traj_point.orientation.w]
            traj[0, i] = traj_point.position.x
            traj[1, i] = traj_point.position.y
            traj[2, i] = ConvertQuaternionToYaw(quaternion)
            traj[3, i] = data.points[i].longitudinal_velocity_mps
        return traj
    elif data_type == "odometry":
        odom = np.zeros((4, 1))
        pose = data.pose.pose
        position = [pose.position.x, pose.position.y]
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        odom[0, 0] = position[0]
        odom[1, 0] = position[1]
        odom[2, 0] = ConvertQuaternionToYaw(quaternion)
        odom[3, 0] = data.twist.twist.linear.x
        return odom
    elif data_type == "goal_pose":
        goal_pose = np.zeros((4, 1))
        pose = data.pose
        position = [pose.position.x, pose.position.y]
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        goal_pose[0, 0] = position[0]
        goal_pose[1, 0] = position[1]
        goal_pose[2, 0] = ConvertQuaternionToYaw(quaternion)
        goal_pose[3, 0] = 0
        return goal_pose
    elif data_type == "obstacle":
        obstacle_list = []
        for obj in data.objects:
            pose = obj.kinematics.initial_pose_with_covariance
            position = [pose.pose.position.x, pose.pose.position.y]
            quaternion = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                          pose.pose.orientation.w]
            center_pose = [position[0], position[1], ConvertQuaternionToYaw(quaternion)]
            contour_shape = [obj.shape.dimensions.x, obj.shape.dimensions.y]
            obstacle = ExtractRectangularContourPoints(center_pose, contour_shape)
            obstacle_list.append(obstacle)
        return obstacle_list
    elif data_type == "map":
        pass
    elif data_type == "numpy":
        N = data.shape[1]
        traj = Trajectory()
        traj.header.stamp = node.get_clock().now().to_msg()
        traj.header.frame_id = "map"
        for i in range(N):
            tp = TrajectoryPoint()
            tp.pose.position.x = data[0, i]
            tp.pose.position.y = data[1, i]
            tp.pose.position.z = 0.0
            quat = ConvertYawToQuaternion(data[2, i])
            tp.pose.orientation.x = quat[0]
            tp.pose.orientation.y = quat[1]
            tp.pose.orientation.z = quat[2]
            tp.pose.orientation.w = quat[3]
            tp.longitudinal_velocity_mps = data[3, i]
            traj.points.append(tp)
        return traj


def IsDataReady():
    global raw_odom, raw_goal, raw_traj, raw_map, raw_acc, raw_steer
    if raw_odom is None or raw_goal is None or raw_traj is None or raw_map is None:
        return False
    if raw_acc is None:
        init_acc = AccelWithCovarianceStamped()
        init_acc.accel.accel.x = 0
        raw_acc = init_acc
    if raw_steer is None:
        init_steer = SteeringReport()
        init_steer.steering_tire_angle = 0
        raw_steer = init_steer
    return True


def UpdateTargetIndex(seg_idx_list):
    global raw_odom, odom_buffer, start_index, end_index, all_traj, sidx
    has_finish = False
    if start_index is None or end_index is None:
        start_index = seg_idx_list[0][0]
        end_index = seg_idx_list[0][1]
        sidx = 0
    p1 = [raw_odom.pose.pose.position.x, raw_odom.pose.pose.position.y]
    p2 = [all_traj.points[end_index].pose.position.x, all_traj.points[end_index].pose.position.y]
    if CalcDistance(p1, p2) < 1.0 and HasStop(odom_buffer):
        sidx += 1
        if sidx <= len(seg_idx_list) - 1:
            start_index = seg_idx_list[sidx][0]
            end_index = seg_idx_list[sidx][1]
            has_finish = False
        else:
            has_finish = True
    return start_index, end_index, has_finish


def Preprocessor2(traj, robot):
    x_list = traj[0, :].tolist()
    y_list = traj[1, :].tolist()
    yaw_list = traj[2, :].tolist()
    vel_list = traj[3, :].tolist()
    # 去除重复节点
    for i in range(len(x_list) - 1, 0, -1):
        if (x_list[i] == x_list[i - 1] and y_list[i] == y_list[i - 1]):
            x_list.pop(i)
            y_list.pop(i)
            yaw_list.pop(i)
            vel_list.pop(i)

    sign = None
    p1 = [x_list[0], y_list[0]]
    p2 = [x_list[1], y_list[1]]
    if (math.fabs(AngleDiff(math.atan2((y_list[1] - y_list[0]), (x_list[1] - x_list[0])), robot[2])) < np.pi / 2):
        sign = 1
        yaw_list[0] = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        vel_list[0] = sign * 1.3888
    else:
        sign = -1
        yaw_list[0] = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        vel_list[0] = sign * 1.3888
    idx = 0
    seg_idx_list = []
    for i in range(1, len(x_list) - 1):
        p1 = [x_list[i - 1], y_list[i - 1]]
        p2 = [x_list[i], y_list[i]]
        p3 = [x_list[i + 1], y_list[i + 1]]

        if (IsShiftPoint(p2, p1, p3)):
            sign = -1 * sign
            seg_idx_list.append((idx, i))
            idx = i

        if sign == 1:
            yaw_list[i] = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
        else:
            yaw_list[i] = math.atan2(p2[1] - p3[1], p2[0] - p3[0])
        vel_list[i] = sign * 1.388
    yaw_list[-1] = yaw_list[-2]
    vel_list[-1] = vel_list[-2]
    seg_idx_list.append((idx, len(x_list) - 1))

    new_traj = np.vstack((x_list, y_list, yaw_list, vel_list))
    return new_traj, seg_idx_list


def Preprocessor(traj, robot):
    x_list = traj[0, :].tolist()
    y_list = traj[1, :].tolist()
    yaw_list = traj[2, :].tolist()
    vel_list = traj[3, :].tolist()
    # 去除重复节点
    for i in range(len(x_list) - 1, 0, -1):
        if (x_list[i] == x_list[i - 1] and y_list[i] == y_list[i - 1]):
            x_list.pop(i)
            y_list.pop(i)
            yaw_list.pop(i)
            vel_list.pop(i)
    sign = None
    if (math.fabs(AngleDiff(math.atan2((y_list[1] - y_list[0]), (x_list[1] - x_list[0])), robot[2])) < np.pi / 2):
        sign = 1
        vel_list[0] = sign * 1.3888
    else:
        sign = -1
        vel_list[0] = sign * 1.3888
    idx = 0
    seg_idx_list = []
    for i in range(1, len(x_list) - 1):
        p1 = [x_list[i - 1], y_list[i - 1]]
        p2 = [x_list[i], y_list[i]]
        p3 = [x_list[i + 1], y_list[i + 1]]

        if (IsShiftPoint(p2, p1, p3)):
            sign = -1 * sign
            seg_idx_list.append((idx, i))
            idx = i
        vel_list[i] = sign * 1.388

    new_traj = np.vstack((x_list, y_list, yaw_list, vel_list))
    return new_traj, seg_idx_list


def GetSegmentIndex(traj, robot):
    x_list = traj[0, :]
    y_list = traj[1, :]
    sign = None
    sign_list = []
    if (math.fabs(AngleDiff(math.atan2((y_list[1] - y_list[0]), (x_list[1] - x_list[0])), robot[2])) < np.pi / 2):
        sign = 1
    else:
        sign = -1
    sign_list.append(sign)
    idx = 0
    seg_idx_list = []
    for i in range(1, len(x_list) - 1):
        p1 = [x_list[i - 1], y_list[i - 1]]
        p2 = [x_list[i], y_list[i]]
        p3 = [x_list[i + 1], y_list[i + 1]]

        if (IsShiftPoint(p2, p1, p3)):
            sign = -1 * sign
            seg_idx_list.append((idx, i))
            idx = i
        sign_list.append(sign)
    seg_idx_list.append((idx, len(x_list) - 1))
    return seg_idx_list, sign_list


def plot_traj(traj):
    posearray = PoseArray()
    posearray.header.frame_id = 'map'
    posearray.header.stamp = node.get_clock().now().to_msg()
    for i in range(len(traj.points)):
        p = traj.points[i].pose
        posearray.poses.append(p)
    plot_pub2 = node.create_publisher(PoseArray, "/obca_traj", 1)
    plot_pub2.publish(posearray)


def plot_traj2(traj):
    # 绘制轨迹点
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.1
    marker.color.a = 1.0
    marker.color.g = 1.0

    traj_points = []
    for i in range(len(traj.points)):
        point = Point()
        point.x = traj.points[i].pose.position.x
        point.y = traj.points[i].pose.position.y
        point.z = 0.0
        traj_points.append(point)
    marker.points = traj_points
    plot_pub = node.create_publisher(Marker, "/obca_marker", 1)
    plot_pub.publish(marker)


def GetSegmentTrajectory(trajecotry, start_index, end_index):
    seg_traj = Trajectory()
    seg_traj.header.stamp = node.get_clock().now().to_msg()
    seg_traj.header.frame_id = 'map'

    lp = Pose()
    lp.position.x = trajectory[0, start_index]
    lp.position.y = trajectory[1, start_index]
    lp.position.z = 0.0
    for i in range(start_index + 1, end_index):
        tpnt = TrajectoryPoint()
        tpnt.longitudinal_velocity_mps = np.sign(trajecotry[3, start_index + 1]) * 1.3888888359069824
        cp = Pose()
        cp.position.x = trajectory[0, i]
        cp.position.y = trajectory[1, i]
        cp.position.z = 0.0

        dx = cp.position.x - lp.position.x
        dy = cp.position.y - lp.position.y
        yaw = math.atan2(dy, dx)
        quat = ConvertYawToQuaternion(yaw)

        lp.orientation.x = quat[0]
        lp.orientation.y = quat[1]
        lp.orientation.z = quat[2]
        lp.orientation.w = quat[3]

        tpnt.pose = lp
        lp = cp
        seg_traj.points.append(tpnt)
    return seg_traj

def GetSegmentTrajectory2(trajectory, start_index, end_index):
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
def OnTimer():
    global ego, all_traj, start_index, end_index, has_new_traj, seg_idx_list, trajectory, traj_pub
    if ego is None:
        ego = GetVehicleInfo()
    if not IsDataReady():
        return

    if has_new_traj:
        start_index = None
        end_index = None
        x0 = ConvertDataFormat("odometry", raw_odom)
        xF = ConvertDataFormat("goal_pose", raw_goal)
        u0 = np.array([[raw_acc.accel.accel.linear.x], [raw_steer.steering_tire_angle]])
        ref_path = ConvertDataFormat("trajectory", raw_traj)
        ref_input = CalculateReferenceInput(ref_path)
        obstacles = ConvertDataFormat("obstacle", raw_obs)
        XYbounds = [-np.inf, np.inf, -np.inf, np.inf]  # map = ConvertDataFormat("map", raw_map)
        trajectory, input = planning(x0, xF, u0, ego, XYbounds, obstacles, ref_path, ref_input, dt=0.5)
        # trajectory, _ = Preprocessor(trajectory, x0)
        # seg_idx_list, _ = GetSegmentIndex(trajectory, x0)
        trajectory, seg_idx_list = Preprocessor2(trajectory, x0)

        all_traj = ConvertDataFormat("numpy", trajectory)
        has_new_traj = False

    start_index, end_index, has_finish = UpdateTargetIndex(seg_idx_list)
    # seg_traj = GetSegmentTrajectory(trajectory, start_index, end_index)
    seg_traj = GetSegmentTrajectory2(trajectory, start_index, end_index)
    traj_pub.publish(seg_traj)
    plot_traj(seg_traj)
    plot_traj2(all_traj)


def GetVehicleInfo():
    global node
    max_vel = node.declare_parameter('vel_lim', 50.0).value
    max_steer = node.declare_parameter('steer_lim', 1.0).value
    max_acc = node.declare_parameter('vel_rate_lim', 7.0).value
    max_steer_rate = node.declare_parameter('steer_rate_lim', 5.0).value
    wheel_base = node.declare_parameter('wheel_base', 1.9).value
    front_overhang = node.declare_parameter('front_overhang', 0.32).value
    rear_overhang = node.declare_parameter('rear_overhang', 0.32).value
    vehicle_length = wheel_base + front_overhang + rear_overhang
    wheel_tread = node.declare_parameter('wheel_tread', 1.465).value
    left_overhang = node.declare_parameter('left_overhang', 0.0).value
    right_overhang = node.declare_parameter('right_overhang', 0.0).value
    vehicle_width = wheel_tread + left_overhang + right_overhang
    rear2center = vehicle_length / 2.0 - rear_overhang

    vehicle_info = {
        "max_vel": max_vel,
        "max_steer": max_steer,
        "max_acc": max_acc,
        "max_steer_rate": max_steer_rate,
        "wheel_base": wheel_base,
        "width": vehicle_width,
        "length": vehicle_length,
        "rear_overhang": rear_overhang,
        "rear2center": rear2center
    }
    return vehicle_info


def main(args=None):
    global node, traj_pub
    rclpy.init(args=args)
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
    map_sub = node.create_subscription(HADMapBin, '/map/vector_map', OnMapCallBack, QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1
    ))

    traj_sub = node.create_subscription(Trajectory, '/planning/scenario_planning/parking/hyridAStar/trajectory',
                                        OnTrajecotryCallBack, rclpy.qos.qos_profile_system_default)
    obs_sub = node.create_subscription(PredictedObjects, '/perception/object_recognition/objects', OnObstacleCallBack,
                                       10)
    odom_sub = node.create_subscription(Odometry, '/localization/kinematic_state', OnOdometryCallBack, 10)
    goal_pose_sub = node.create_subscription(PoseStamped, '/planning/mission_planning/goal', OnGoalCallBack, 10)
    acc_sub = node.create_subscription(AccelWithCovarianceStamped, '/localization/acceleration', OnAcclerationCallBack,
                                       10)
    steer_sub = node.create_subscription(SteeringReport, '/vehicle/status/steering_status', OnSteeringCallBack, 10)
    traj_pub = node.create_publisher(Trajectory, '/planning/scenario_planning/parking/trajectory',
                                     rclpy.qos.qos_profile_system_default)

    timer_period = 0.1
    timer = node.create_timer(timer_period, OnTimer)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
