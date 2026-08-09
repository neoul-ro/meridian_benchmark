from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # Placeholder (plan: TBD). Scored against its own input (invariants:
    # monotonic version, unique ids, commit consistency) — recorder keeps
    # arrivals; payload capture comes with the backend scorer.
    return bench_launch(
        'graphcore',
        player_topics=[], expected_subs=[],
        record=['/object_update_set=update_set',
                '/local_object_graph_snapshot=snapshot',
                '/graph_update_event=event'],
        payload=[],
        module_pkg='meridian_graphcore', module_exe='graphcore_node',
        notes=['[bench] graphcore: placeholder — invariant scoring TBD'])
