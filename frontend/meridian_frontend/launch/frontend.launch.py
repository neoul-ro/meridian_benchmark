#!/usr/bin/env python3
# frontend.launch.py - 조립된 frontend 프로세스 1개 실행. 토픽·엔진 경로 인자 전달만 담당.
# 노드 이름은 코드(assemble)가 소유한다 - 여기서 name= 을 주면 노드 이름이 덮여 출력 토픽 경로가 바뀐다.
# 모듈 조립은 frontend.py 내부. 프로세스 분리 불가(Tegra CUDA IPC consumer 미지원).
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    engines = os.path.join(get_package_share_directory('meridian_frontend'), 'engines')   # setup.py data_files
    declared = [
        DeclareLaunchArgument('rgb_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument('info_topic', default_value='/camera/color/camera_info'),
        DeclareLaunchArgument('pose_topic', default_value='/pose'),     # PoseStamped, 카메라 world pose
        DeclareLaunchArgument('tracklet_topic', default_value='~/tracklet_set'),  # 빈 문자열이면 publish 없음
        DeclareLaunchArgument('seg_engine', default_value=os.path.join(engines, 'fastsam.plan')),
        DeclareLaunchArgument('clip_engine', default_value=os.path.join(engines, 'clip_image.plan')),
    ]
    return LaunchDescription(declared + [
        Node(
            package='meridian_frontend', executable='frontend', output='screen',
            arguments=[
                '--rgb-topic', LaunchConfiguration('rgb_topic'),
                '--depth-topic', LaunchConfiguration('depth_topic'),
                '--info-topic', LaunchConfiguration('info_topic'),
                '--pose-topic', LaunchConfiguration('pose_topic'),
                '--tracklet-topic', LaunchConfiguration('tracklet_topic'),
                '--seg-engine', LaunchConfiguration('seg_engine'),
                '--clip-engine', LaunchConfiguration('clip_engine'),
            ],
        ),
    ])
