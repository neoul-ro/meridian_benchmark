from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # No launchable rgb-d SLAM executable exists upstream (meridian_slam is
    # FAST-LIVO2, C++ and lidar/imu-driven), so this starts only the harness:
    # player (rgb/depth/info) + recorder (/pose). Start the SLAM under test
    # separately, then it will pick up the camera topics.
    return bench_launch(
        'slam',
        player_topics=['rgb', 'depth', 'info'], expected_subs=[1, 1, 1],
        record=['/camera/rgb=image', '/camera/depth=image', '/pose=pose'],
        payload=[],
        notes=['[bench] slam: start the SLAM module separately; '
               'recording /pose as geometry_msgs/PoseStamped'])
