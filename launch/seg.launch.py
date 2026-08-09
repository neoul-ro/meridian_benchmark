from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # rgb: seg module + recorder
    return bench_launch(
        'seg',
        player_topics=['rgb'], expected_subs=[2],
        record=['/camera/rgb=image', '/segment_image=image'],
        payload=['/segment_image'],
        module_pkg='meridian_seg', module_exe='seg_node')
