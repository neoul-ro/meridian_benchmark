from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # rgb/seg: clip module + recorder
    return bench_launch(
        'clip',
        player_topics=['rgb', 'seg'], expected_subs=[2, 2],
        record=['/camera/rgb=image', '/segment_image=image',
                '/instance_embedding_set=embedding_set'],
        payload=['/instance_embedding_set'],
        module_pkg='meridian_clip', module_exe='clip_inference_node')
