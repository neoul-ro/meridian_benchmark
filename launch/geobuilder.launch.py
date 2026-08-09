from launch.substitutions import LaunchConfiguration, PythonExpression

from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # depth/seg/pose: geobuilder + recorder; info: geobuilder only.
    # /pose is world_T_base (PoseStamped); until the geobuilder composes the
    # base_T_camera extrinsic itself, run with pose_source:=cam to feed the
    # camera pose directly. While geobuilder still subscribes
    # PoseWithCovarianceStamped, pass pose_type:=cov to match it
    # (the recorder's /pose subscription follows pose_type).
    pose_record = PythonExpression(
        ["'/pose=pose_cov' if '", LaunchConfiguration('pose_type'),
         "' == 'cov' else '/pose=pose'"])
    return bench_launch(
        'geobuilder',
        player_topics=['depth', 'info', 'seg', 'pose'],
        expected_subs=[2, 1, 2, 2],
        record=['/camera/depth=image', '/segment_image=image',
                pose_record, '/instance_3d_set=instance3d_set'],
        payload=['/instance_3d_set'],
        module_pkg='meridian_geobuilder', module_exe='geobuilder_node')
