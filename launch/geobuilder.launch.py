from launch.substitutions import LaunchConfiguration, PythonExpression

from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # depth/seg/pose: geobuilder + recorder; info: geobuilder only.
    # Upstream geobuilder still subscribes PoseWithCovarianceStamped and uses
    # /pose directly as world_T_camera (no base_T_camera extrinsic param), so
    # default to pose_type:=cov + pose_source:=cam until upstream matches the
    # wiki contract (the recorder's /pose subscription follows pose_type).
    pose_record = PythonExpression(
        ["'/pose=pose_cov' if '", LaunchConfiguration('pose_type'),
         "' == 'cov' else '/pose=pose'"])
    return bench_launch(
        'geobuilder',
        player_topics=['depth', 'info', 'seg', 'pose'],
        expected_subs=[2, 1, 2, 2],
        pose_source_default='cam', pose_type_default='cov',
        record=['/camera/depth=image', '/segment_image=image',
                pose_record, '/instance_3d_set=instance3d_set'],
        payload=['/instance_3d_set'],
        module_pkg='meridian_geobuilder', module_exe='geobuilder_node')
