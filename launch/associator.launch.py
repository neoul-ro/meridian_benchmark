from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # Placeholder. GT tracklets exist as <gt>/gt_tracklets.h5 (one Tracklet
    # per visibility episode, see gt_tracklets.py) but the wire topic/type for
    # feeding them to the associator is still undecided, so nothing is
    # injected here yet; arrivals of the backend topics are recorded.
    return bench_launch(
        'associator',
        player_topics=[], expected_subs=[],
        record=['/tracklet_set=tracklet_set',
                '/association_decision_set=decision_set',
                '/local_object_graph_snapshot=snapshot'],
        payload=[],
        module_pkg='meridian_associator', module_exe='associator_node',
        notes=['[bench] associator: placeholder - GT tracklet feed topic TBD'])
