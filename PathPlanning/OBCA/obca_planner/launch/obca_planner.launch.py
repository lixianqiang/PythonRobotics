
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    vehicle_info_param_path = '/home/stephen_li/autoware/src/vehicle/sample_vehicle_launch/sample_vehicle_description/config/vehicle_info.param.yaml'
    # vehicle_info_param_path = '/home/用户名/autoware/src/vehicle/sample_vehicle_launch/sample_vehicle_description/config/vehicle_info.param.yaml'

    with open(vehicle_info_param_path, "r") as f:
        vehicle_info_param = yaml.safe_load(f)["/**"]["ros__parameters"]

    return LaunchDescription(
        [Node(
            package='obca_planner',
            executable='obca_planner_node',
            output='screen',
            namespace='/planning/scenario_planning/parking/',
            remappings=[
                ('~/input/map', '/map/vector_map'),
                ('~/input/goal_pose', '/planning/mission_planning/goal'),
                ('~/input/trajectory', '/planning/scenario_planning/parking/hyridAStar/trajectory'),
                ('~/input/obstacles', '/perception/object_recognition/objects'),
                ("~/input/odometry", "/localization/kinematic_state"),
                ("~/input/acceleration", "/localization/acceleration"),
                ("~/input/steering", "/vehicle/status/steering_status"),
                ("~/output/trajectory", "/planning/scenario_planning/parking/trajectory")
            ],
            parameters=[vehicle_info_param]
        )]
    )