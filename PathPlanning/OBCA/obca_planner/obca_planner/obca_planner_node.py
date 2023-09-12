import rclpy
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_auto_perception_msgs.msg import PredictedObjects
from autoware_auto_vehicle_msgs.msg import SteeringReport
from autoware_auto_mapping_msgs.msg import HADMapBin
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point
from geometry_msgs.msg import AccelWithCovarianceStamped
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


def IsDiffTraj(traj1: Trajectory, traj2: Trajectory):
    if len(traj1.points) != len(traj2.points):
        return True
    for p1, p2 in zip(traj1.points, traj2.points):
        if IsDiffPose(p1.pose, p2.pose, 0.25, np.pi / 18):
            return True
    return False


def OnMapCallBack(msg):
    global raw_data
    raw_data['map'] = msg


error_val = None


def OnTrajecotryCallBack(msg: Trajectory):
    global raw_data, has_new_traj, has_new_goal, error_val
    if len(msg.points) <= 1 or raw_data['goal'] is None: return
    if raw_data['traj'] is None:
        raw_data['traj'] = msg
        has_new_traj = True
        return
    elif len(raw_data['traj'].points) != len(msg.points) or IsDiffPose(raw_data['traj'].points[-1].pose,
                                                                       msg.points[-1].pose, 1e-2, 1e-2):
        has_new_traj = True
        raw_data['traj'] = msg
        return
    if has_new_traj:
        p1 = raw_data['goal'].pose.position.x, raw_data['goal'].pose.position.y
        p2 = raw_data['traj'].points[-1].pose.position.x, raw_data['traj'].points[-1].pose.position.y
        if error_val is None:
            error_val = CalcDistance(p1, p2)
        elif math.fabs(CalcDistance(p1, p2) - error_val) > 1e-3:
            error_val = CalcDistance(p1, p2)


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
    if raw_data['goal'] is None or IsDiffPose(raw_data['goal'].pose, msg.pose, 1e-3, 1e-3):
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


def CalculateReferenceInput(ref_path, u0=[0.0, 0.0]):
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
    steer_list.append(0.0) # steer_list.append(steer_list[-1])
    # acc_list = [0 for i in range(len(steer_list))]
    acc_list = []
    for j in range(1, len(x_list) - 1):
        if (vel_list[j - 1] * vel_list[j] < 0):
            acc_list.append(0)
        else:
           acc_list.append(vel_list[j] - vel_list[j - 1])
    acc_list.append(0)
    ref_input = np.vstack((acc_list, steer_list))
    return ref_input


def ConvertDataFormat(node, data_type, data):
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


def IsDataReady(raw_data):
    if raw_data['odom'] is None or raw_data['goal'] is None or raw_data['traj'] is None or raw_data['map'] is None:
        return False
    return True


def UpdateTargetIndex(trajectory, seg_idx_list, start_index, end_index):
    global raw_data, odom_buffer, sidx
    has_finish = False
    if start_index is None or end_index is None:
        start_index = seg_idx_list[0][0]
        end_index = seg_idx_list[0][1]
        sidx = 0
    p1 = [raw_data['odom'].pose.pose.position.x, raw_data['odom'].pose.pose.position.y]
    p2 = [trajectory[0, end_index], trajectory[1, end_index]]
    if CalcDistance(p1, p2) < 1.0 and HasStop(odom_buffer):
        sidx += 1
        if sidx <= len(seg_idx_list) - 1:
            start_index = seg_idx_list[sidx][0]
            end_index = seg_idx_list[sidx][1]
            has_finish = False
        else:
            has_finish = True
    return start_index, end_index, has_finish


def Postprocessor(traj, robot):
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
start_index = None
end_index = None
odom_buffer = []
sidx = None
raw_data = {
    "traj": None,
    "odom": None,
    "goal": None,
    "obs": None,
    "acc": None,
    "steer": None,
    "map": None
}

ref_path = None
ref_input = None

use_txt = False
def OnTimer(node):
    global ego, traj_pub, raw_data, has_new_traj, trajectory, seg_idx_list, start_index, end_index, ref_path, ref_input
    if ego is None:
        ego = GetVehicleInfo(node)
    if traj_pub is None:
        traj_pub = node.create_publisher(Trajectory, '/planning/scenario_planning/parking/trajectory',
                                         rclpy.qos.qos_profile_system_default)
    global use_txt
    if use_txt:
        start_index = None
        end_index = None
        x0, xF, u0, ref_path, ref_input, raw_ref_path = RecoverAllData()
        obstacles = ConvertDataFormat(node, "obstacle", raw_data['obs'])
        XYbounds = [-np.inf, np.inf, -np.inf, np.inf]  # map = ConvertDataFormat(node, "map", raw_map)
        trajectory, input = planning(x0, xF, u0, ego, XYbounds, obstacles, ref_path, ref_input, dt=0.5)
        trajectory, seg_idx_list = Postprocessor(trajectory, x0)
        has_new_traj = False
    else:
        if not IsDataReady(raw_data):
            return
        if has_new_traj:
            start_index = None
            end_index = None
            x0 = ConvertDataFormat(node, "odometry", raw_data['odom'])
            xF = ConvertDataFormat(node, "goal_pose", raw_data['goal'])
            u0 = np.array([[raw_data['acc'].accel.accel.linear.x], [raw_data['steer'].steering_tire_angle]])
            ref_path = ConvertDataFormat(node, "trajectory", raw_data['traj'])

            ref_input = CalculateReferenceInput(ref_path, u0)
            obstacles = ConvertDataFormat(node, "obstacle", raw_data['obs'])
            XYbounds = [-np.inf, np.inf, -np.inf, np.inf]  # map = ConvertDataFormat(node, "map", raw_map)
            trajectory, input = planning(x0, xF, u0, ego, XYbounds, obstacles, ref_path, ref_input, dt=0.5)
            trajectory, seg_idx_list = Postprocessor(trajectory, x0)
            has_new_traj = False

    if trajectory is not None and seg_idx_list != []:
        start_index, end_index, has_finish = UpdateTargetIndex(trajectory, seg_idx_list, start_index, end_index)

        all_traj = GetSegmentTrajectory(trajectory, 0, trajectory.shape[1])
        seg_traj = GetSegmentTrajectory(trajectory, start_index, end_index)
        ref_path1 = GetSegmentTrajectory(ref_path, 0, ref_path.shape[1])

        traj_pub.publish(seg_traj)
        plot_arrow(all_traj, "/obca_all_traj_ori")
        plot_arrow(seg_traj, "/obca_seg_traj_ori")
        plot_arrow(ref_path1, "/obca_ref_path_ori")
        plot_line(all_traj, "/obca_all_traj")
        plot_line(seg_traj, "/obca_seg_traj")
        plot_line(ref_path1, "/obca_ref_path")

        temp_yaw = []
        temp_yaw_car = []
        temp_yaw_ref_input2 = []
        ref_path2 = np.zeros(ref_path.shape)
        for i in range(1, ref_input.shape[1]):
            ref_path2[0, i] = ref_path[0, i]
            ref_path2[1, i] = ref_path[1, i]
            ref_path2[2, i] = ref_input[1, i - 1] + ref_path[2, i]
            temp_yaw.append(np.rad2deg(ref_input[1, i - 1]))
            temp_yaw_car.append(np.rad2deg(ref_path[2, i]))
            temp_yaw_ref_input2.append(np.rad2deg(ref_input[1, i - 1] + ref_path[2, i]))
        ref_path21 = GetSegmentTrajectory(ref_path2, 0, ref_path2.shape[1])
        plot_arrow(ref_path21, "/obca_ref_input_ori")


def GetVehicleInfo(node):
    max_vel = node.declare_parameter('vel_lim', 10.0).value  # robobus 10.0 sample 50.0
    max_steer = node.declare_parameter('steer_lim', 1.0).value  # robobus 1.0 sample 1.0
    max_acc = node.declare_parameter('vel_rate_lim', 3.0).value  # robobus 2.0 sample 7.0
    max_steer_rate = node.declare_parameter('steer_rate_lim', 0.5).value  # robobus 5.0 sample 5.0
    wheel_base = node.declare_parameter('wheel_base', 3.02).value  # robobus 3.020 sample 1.9
    front_overhang = node.declare_parameter('front_overhang', 0.32).value  # robobus 0.400 sample 0.32
    rear_overhang = node.declare_parameter('rear_overhang', 0.32).value  # robobus 0.400 sample 0.32
    vehicle_length = wheel_base + front_overhang + rear_overhang
    wheel_tread = node.declare_parameter('wheel_tread', 1.465).value  # robobus 1.618 sample 1.465
    left_overhang = node.declare_parameter('left_overhang', 0.0).value  # robobus 0.0 sample 0.0
    right_overhang = node.declare_parameter('right_overhang', 0.0).value  # robobus 0.0 sample 0.0
    vehicle_width = wheel_tread + left_overhang + right_overhang
    rear2center = vehicle_length / 2.0 - rear_overhang # rear2center = 0

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
    node.get_logger().info("xxxxx wheel_base is {:.3f}".format(wheel_base))
    node.get_logger().info("xxxxx front_overhang is {:.3f}".format(front_overhang))
    node.get_logger().info("xxxxx rear_overhang is {:.3f}".format(rear_overhang))
    node.get_logger().info("xxxxx wheel_tread is {:.3f}".format(wheel_tread))
    node.get_logger().info("xxxxx left_overhang is {:.3f}".format(left_overhang))
    node.get_logger().info("xxxxx right_overhang is {:.3f}".format(right_overhang))
    return vehicle_info

record = False
def OnTimer4Record(node):
    global record
    if record:  # 在debug程序的时候手动设置为True,然后当程序退出后会生成对应文件
        def RecordAllData(node, raw_data):
            def write_array_to_text(array, file_path):
                with open(file_path, 'w') as file:
                    np.savetxt(file, array, delimiter=',')

            x0 = ConvertDataFormat(node, "odometry", raw_data['odom'])
            xF = ConvertDataFormat(node, "goal_pose", raw_data['goal'])
            u0 = np.array([[raw_data['acc'].accel.accel.linear.x], [raw_data['steer'].steering_tire_angle]])
            ref_path = ConvertDataFormat(node, "trajectory", raw_data['traj'])
            ref_input = CalculateReferenceInput(ref_path)
            raw_ref_path = np.zeros((7, len(raw_data['traj'].points)))
            for i in range(len(raw_data['traj'].points)):
                raw_ref_path[0, i] = raw_data['traj'].points[i].pose.position.x
                raw_ref_path[1, i] = raw_data['traj'].points[i].pose.position.y
                raw_ref_path[2, i] = raw_data['traj'].points[i].pose.orientation.x
                raw_ref_path[3, i] = raw_data['traj'].points[i].pose.orientation.y
                raw_ref_path[4, i] = raw_data['traj'].points[i].pose.orientation.z
                raw_ref_path[5, i] = raw_data['traj'].points[i].pose.orientation.w
                raw_ref_path[6, i] = raw_data['traj'].points[i].longitudinal_velocity_mps
            write_array_to_text(x0, "x0.txt")
            write_array_to_text(xF, "xF.txt")
            write_array_to_text(u0, "u0.txt")
            write_array_to_text(ref_path, "ref_path.txt")
            write_array_to_text(ref_input, "ref_input.txt")
            write_array_to_text(raw_ref_path, "raw_ref_path.txt")
        global raw_data, ego
        RecordAllData(node, raw_data)
        record = False

pub_init = False
pub_goal = False
def OnTimer4Init_Goal(node):
    global pub_init
    if pub_init:
        from geometry_msgs.msg import PoseWithCovarianceStamped
        init_pose_pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        init_pose = PoseWithCovarianceStamped()
        p = Pose()
        p.position.x = 3.243661880493164062e+01
        p.position.y = -5.368216514587402344e+00
        target_yaw = -3.132069106700981376e+00
        quat = ConvertYawToQuaternion(target_yaw)
        p.orientation.x = quat[0]
        p.orientation.y = quat[1]
        p.orientation.z = quat[2]
        p.orientation.w = quat[3]

        init_pose.pose.pose = p
        init_pose.header.frame_id = "map"
        init_pose.header.stamp = node.get_clock().now().to_msg()
        init_pose_pub.publish(init_pose)
        pub_init = False
    global pub_goal
    if pub_goal:
        goal_pose_pub = node.create_publisher(PoseStamped, '/planning/mission_planning/goal', 1)
        goal_pose = PoseStamped()
        p = Pose()
        p.position.x = 4.497026062011718750e+01
        p.position.y = -7.964234828948974609e+00
        target_yaw = 1.792877627312906119e+00
        quat = ConvertYawToQuaternion(target_yaw)
        p.orientation.x = quat[0]
        p.orientation.y = quat[1]
        p.orientation.z = quat[2]
        p.orientation.w = quat[3]
        goal_pose.pose = p
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = node.get_clock().now().to_msg()
        goal_pose_pub.publish(goal_pose)
        pub_goal = False

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
    timer4record = node.create_timer(timer_period_sec=0.1, callback=functools.partial(OnTimer4Record, node))
    timer4init_goal = node.create_timer(timer_period_sec=0.1, callback=functools.partial(OnTimer4Init_Goal, node))
    timer = node.create_timer(timer_period_sec=0.1, callback=functools.partial(OnTimer, node))
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
