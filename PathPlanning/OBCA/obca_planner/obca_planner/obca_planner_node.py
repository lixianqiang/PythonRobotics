import sys
import casadi as ca
import numpy as np
import rclpy
import math
from rclpy.node import Node
from autoware_auto_planning_msgs.msg import Trajectory
from autoware_auto_perception_msgs.msg import PredictedObjects
from autoware_auto_vehicle_msgs.msg import SteeringReport
from autoware_auto_mapping_msgs.msg import HADMapBin
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import AccelWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
from obca_planner.obca import *
from obca_planner.unit import *

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

raw_map = None


def OnMapCallBack(msg):
    global raw_map
    raw_map = msg


raw_traj = None


def OnReferenceTrajectoryCallBack(msg: Trajectory):
    global raw_traj
    raw_traj = msg


raw_odom = None


def OnOdometryCallBack(msg):
    global raw_odom
    raw_odom = msg


raw_goal_pose = None


def OnGoalPoseCallBack(msg):
    global raw_goal_pose
    raw_goal_pose = msg


raw_obs = None


def OnObstacleCallBack(msg):
    global raw_obs
    raw_obs = msg


raw_acc = None


def OnAcclerationCallBack(msg):
    global raw_acc
    raw_acc = msg


raw_steer = None


def OnSteeringCallBack(msg):
    global raw_steer
    raw_steer = msg


def CalculateReferenceInput(ref_path):
    steer_list = []
    x_list, y_list, vel_list = ref_path[0], ref_path[1], ref_path[3]
    orientation = vel_list[0]
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


ego = None


def OnTimer():
    global ego, raw_map, raw_acc, raw_steer, raw_traj, raw_odom, raw_obs
    if ego is None:
        ego = GetVehicleInfo()

    if rclpy.ok():
        if raw_traj:
            ref_path = ConvertDataFormat("trajectory", raw_traj)
        if raw_odom:
            x0 = ConvertDataFormat("odometry", raw_odom)
        if raw_goal_pose:
            xF = ConvertDataFormat("goal_pose", raw_goal_pose)
        if raw_acc and raw_steer:
            u0 = np.array([[raw_acc.accel.accel.linear.x], [raw_steer.steering_tire_angle]])
        if raw_obs:
            obstacles = ConvertDataFormat("obstacle", raw_obs)
        if raw_map:
            # map = ConvertDataFormat("map", raw_map)
            XYbounds = [-np.inf, np.inf, -np.inf, np.inf]
        if raw_traj is None or raw_odom is None or raw_goal_pose is None or raw_obs is None or raw_map is None:
            return
        traj_xy = ref_path[:2, -1]
        goal_xy = xF[:2, 0]
        if (np.hypot(traj_xy[0] - goal_xy[0], traj_xy[1] - goal_xy[1]) < 2.0):
            ref_input = CalculateReferenceInput(ref_path)
            # ref_path = np.zeros((4, 51))
            # ref_input = np.zeros((2, 50))
            trajectory, input = planning(x0, xF, u0, ego, XYbounds, obstacles, ref_path, ref_input, dt=0.5)

            # 绘制轨迹点
            global node
            plot_pub = node.create_publisher(Marker, "/obca_marker", 1)
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = node.get_clock().now().to_msg()
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.color.a = 1.0
            marker.color.g = 1.0

            traj_points = []
            for i in range(trajectory.shape[1]):
                point = Point()
                point.x = trajectory[0,i]
                point.y = trajectory[1, i]
                point.z = 0.0
                traj_points.append(point)
            marker.points = traj_points
            plot_pub.publish(marker)


def GetVehicleInfo():
    global  node
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


node = None


def main(args=None):
    global node
    rclpy.init(args=args)
    node = rclpy.create_node('obca_planner_node')
    # 通过obca_planner.launch.py 启动时使用
    # map_sub = node.create_subscription(HADMapBin, '~/input/map', OnMapCallBack, QoSProfile(
    #     reliability=ReliabilityPolicy.RELIABLE,
    #     history=HistoryPolicy.KEEP_LAST,
    #     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    #     depth=1
    # ))
    # traj_sub = node.create_subscription(Trajectory, '~/input/trajectory', OnReferenceTrajectoryCallBack, 10)
    # obs_sub = node.create_subscription(PredictedObjects, '~/input/obstacles', OnObstacleCallBack, 10)
    # odom_sub = node.create_subscription(Odometry, '~/input/odometry', OnOdometryCallBack, 10)
    # goal_pose_sub = node.create_subscription(PoseStamped, '~/input/goal_pose', OnGoalPoseCallBack, 10)
    # acc_sub = node.create_subscription(AccelWithCovarianceStamped, '~/input/acceleration', OnAcclerationCallBack, 10)
    # steer_sub = node.create_subscription(SteeringReport, '~/input/steering', OnSteeringCallBack, 10)

    # 单独启动节点
    map_sub = node.create_subscription(HADMapBin, '/map/vector_map', OnMapCallBack, QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        depth=1
    ))
    
    traj_sub = node.create_subscription(Trajectory, '/planning/scenario_planning/parking/trajectory',
                                        OnReferenceTrajectoryCallBack, 10)
    obs_sub = node.create_subscription(PredictedObjects, '/perception/object_recognition/objects', OnObstacleCallBack,
                                       10)
    odom_sub = node.create_subscription(Odometry, '/localization/kinematic_state', OnOdometryCallBack, 10)
    goal_pose_sub = node.create_subscription(PoseStamped, '/planning/mission_planning/goal', OnGoalPoseCallBack, 10)
    acc_sub = node.create_subscription(AccelWithCovarianceStamped, '/localization/acceleration', OnAcclerationCallBack,
                                       10)
    steer_sub = node.create_subscription(SteeringReport, '/vehicle/status/steering_status', OnSteeringCallBack, 10)

    timer_period = 0.1
    timer = node.create_timer(timer_period, OnTimer)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
