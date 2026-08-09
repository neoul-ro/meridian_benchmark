from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # Injection works (GT instance_3d_set + instance_embedding_set);
    # scoring (tracklet purity / fragmentation) is TBD per the plan.
    return bench_launch(
        'geotracker',
        player_topics=['instance3d', 'embedding'], expected_subs=[2, 2],
        record=['/instance_3d_set=instance3d_set',
                '/instance_embedding_set=embedding_set',
                '/tracklet_set=tracklet_set'],
        payload=[],
        module_pkg='meridian_geotracker', module_exe='geotracker_node',
        notes=['[bench] geotracker: scoring TBD (arrivals recorded only)'])
