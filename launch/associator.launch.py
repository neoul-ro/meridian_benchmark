from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # Placeholder (plan: TBD). No GT injection — the associator consumes real
    # upstream output (/tracklet_set) plus graphcore snapshots; run it
    # together with the upstream modules. Arrivals are recorded for timing.
    return bench_launch(
        'associator',
        player_topics=[], expected_subs=[],
        record=['/tracklet_set=tracklet_set',
                '/association_decision_set=decision_set'],
        payload=[],
        module_pkg='meridian_associator', module_exe='associator_node',
        notes=['[bench] associator: placeholder — needs upstream tracklets '
               'and a running graphcore (snapshot is TRANSIENT_LOCAL)'])
