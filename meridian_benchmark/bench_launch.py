"""Shared builder for the per-module benchmark launch files.

Every launch exposes the same arguments:
  dataset:=<uHumans2 sequence dir>   (player input images)
  gt:=<GT dir for the sequence>      (player input GT artifacts)
  out:=<run output dir>              (recorder + play manifest)
  input:=true|false     start the player (false = inputs come from other
                        modules in a combo run, per the plan)
  module:=true|false    also start the upstream stub of the module under
                        test (default false: launch only the harness and
                        run the module separately — the player waits for
                        its subscription before playback starts)
  rate / start_frame / end_frame / pose_source   forwarded to the player

The player's exit shuts the whole launch down (recorder flushes per row).
"""

import os
import time

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, LogInfo,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

DEFAULT_DATASET = os.path.expanduser(
    '~/yun/meridian_ws/datasets/uHumans2/apartment_scene/'
    'uHumans2_apartment_s1_00h')
DEFAULT_GT = os.path.expanduser(
    '~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h')


def bench_launch(name, player_topics, expected_subs, record, payload,
                 module_pkg=None, module_exe=None, module_default='false',
                 pose_source_default='base', pose_type_default='plain',
                 notes=()):
    dataset = LaunchConfiguration('dataset')
    gt = LaunchConfiguration('gt')
    out = LaunchConfiguration('out')

    actions = [
        DeclareLaunchArgument('dataset', default_value=DEFAULT_DATASET),
        DeclareLaunchArgument('gt', default_value=DEFAULT_GT),
        # timestamped so repeated runs never share a dir; the recorder
        # refuses to overwrite an existing recording
        DeclareLaunchArgument('out', default_value=os.path.join(
            os.path.expanduser('~/yun/meridian_ws/bench_runs'), name,
            time.strftime('%Y%m%d_%H%M%S'))),
        DeclareLaunchArgument('input', default_value='true'),
        DeclareLaunchArgument('module', default_value=module_default),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('start_frame', default_value='0'),
        DeclareLaunchArgument('end_frame', default_value='0'),
        DeclareLaunchArgument('pose_source', default_value=pose_source_default),
        DeclareLaunchArgument('pose_type', default_value=pose_type_default),
    ]
    for msg in notes:
        actions.append(LogInfo(msg=msg))

    # each entry nested in its own list so mixed str/Substitution entries
    # stay a STRING_ARRAY instead of concatenating into one string
    recorder = Node(
        package='meridian_benchmark', executable='recorder',
        name='bench_recorder', output='screen',
        parameters=[{'out': out, 'record': [[e] for e in record],
                     'payload': list(payload) or ['']}])
    actions.append(recorder)

    if module_pkg:
        actions.append(Node(
            package=module_pkg, executable=module_exe, output='screen',
            condition=IfCondition(LaunchConfiguration('module'))))

    if player_topics:
        player = Node(
            package='meridian_benchmark', executable='player',
            name='bench_player', output='screen',
            condition=IfCondition(LaunchConfiguration('input')),
            parameters=[{
                'dataset': dataset, 'gt': gt,
                'topics': list(player_topics),
                'expected_subs': list(expected_subs),
                'rate': LaunchConfiguration('rate'),
                'start_frame': LaunchConfiguration('start_frame'),
                'end_frame': LaunchConfiguration('end_frame'),
                'pose_source': LaunchConfiguration('pose_source'),
                'pose_type': LaunchConfiguration('pose_type'),
                'manifest': PythonExpression(
                    ["'", out, "' + '/play_manifest.json'"]),
            }])
        actions.append(player)
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=player,
            on_exit=[EmitEvent(event=Shutdown(
                reason='player finished'))])))

    return LaunchDescription(actions)
