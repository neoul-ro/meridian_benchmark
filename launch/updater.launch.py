from meridian_benchmark.bench_launch import bench_launch


def generate_launch_description():
    # Placeholder (plan: TBD). Consumes real upstream decisions + snapshots.
    return bench_launch(
        'updater',
        player_topics=[], expected_subs=[],
        record=['/association_decision_set=decision_set',
                '/object_update_set=update_set'],
        payload=[],
        module_pkg='meridian_updater', module_exe='updater_node',
        notes=['[bench] updater: placeholder — needs upstream decisions '
               'and a running graphcore'])
