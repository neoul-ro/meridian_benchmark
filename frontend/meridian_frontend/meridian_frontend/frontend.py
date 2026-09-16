#!/usr/bin/env python3
"""Perception Frontend 단일 프로세스 조립 엔트리.

체인: preprocessing → sam → tracker(+gate) → clip → publisher; pose 입력 = /pose PoseStamped (카메라 world pose).
배선 정의는 assemble() 하나 — launch 와 실험 러너가 공유. 모듈 간 GPU 텐서는 주소 직접 참조(단일 프로세스 전제).
"""
import os
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')   # 엔트리 최상단 필수 (inductor 병렬 컴파일 hang)
import argparse
import sys
import time

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node

ENGINE_FILES = dict(seg='fastsam.plan', clip='clip_image.plan')


def engine_path(kind, given=None):
    """엔진 파일 경로. given 이 비어 있으면 이 패키지의 share/engines/<name> — colcon 이 setup.py data_files 로
    복사해 두므로 src/ 를 어디로 옮겨 빌드해도 경로가 따라온다. 파일이 없으면 조립 전에 멈춘다."""
    path = given
    if not path:
        try:
            share = get_package_share_directory('meridian_frontend')
        except PackageNotFoundError:
            raise SystemExit('meridian_frontend 미설치 — colcon build 후 install/setup.bash 를 source 할 것')
        path = os.path.join(share, 'engines', ENGINE_FILES[kind])
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise SystemExit(f'{kind} 엔진 없음: {path} — src/meridian_frontend/engines/ 에 {ENGINE_FILES[kind]} 을 두고 '
                         f'colcon build 하거나 --{kind}-engine 으로 경로를 줄 것. 경로가 엉뚱한 WS 를 가리키면 '
                         f'의도한 WS 의 install/setup.bash 를 마지막에 source 했는지 확인(같은 이름의 다른 패키지가 앞설 수 있음)')
    return path


def parse_args(argv=None):
    """CLI 인자 → cfg. --ros-args 공존을 위해 parse_known_args. 엔진 경로는 여기서 확정(부재 시 즉시 종료)."""
    ap = argparse.ArgumentParser(prog='frontend', add_help=False)
    ap.add_argument('-h', '--help', action='help')
    ap.add_argument('--rgb-topic', default='/camera/color/image_raw')
    ap.add_argument('--depth-topic', default='/camera/aligned_depth_to_color/image_raw')
    ap.add_argument('--info-topic', default='/camera/color/camera_info')
    ap.add_argument('--pose-topic', default='/pose',
                    help='geometry_msgs/PoseStamped, 카메라 optical frame 의 world pose')
    ap.add_argument('--seg-engine', default='', help='빈 문자열이면 패키지 share/engines/fastsam.plan')
    ap.add_argument('--clip-engine', default='', help='빈 문자열이면 패키지 share/engines/clip_image.plan')
    ap.add_argument('--tracklet-topic', default='~/tracklet_set',
                    help='private 이름 → /<node>/tracklet_set 으로 해석 · 빈 문자열이면 publish 없음')
    ap.add_argument('--nov-tau', type=float, default=None, help='coverage 발화 임계 (기본 0.20)')
    cfg, _ = ap.parse_known_args(argv)
    cfg.seg_engine = engine_path('seg', cfg.seg_engine)
    cfg.clip_engine = engine_path('clip', cfg.clip_engine)
    return cfg


def assemble(cfg, da_sink=None):
    """파이프라인 배선의 유일한 정의. 반환 = (nodes, ctx).

    노드는 frontend 1개. 모듈은 이 노드에 구독·발행을 등록하는 평범한 클래스다.

    da_sink 를 주면 발화 결과(TrackletSetPayload)를 그 함수로 직접 넘긴다 — 같은 프로세스에 DA 를
    올렸을 때 직렬화 없이 받는 경로. 계약 4가지는 publisher.py 상단에 전문이 있고 요지는 이렇다.
      1. 호출 스레드 = publisher-worker (executor 스레드 아님).
      2. payload 의 tracklet_ids·points·depth_valid_ratio·seg_conf 는 pinned view 라 다음 발화가 덮는다.
         보관하려면 복사할 것. 나머지 3종(points_num·id_emb·query_emb)은 복사본이다.
      3. view 는 읽기 전용. 제자리 수정은 예외가 난다.
      4. 시간 예산 = 발화 간격(약 167ms). 넘으면 입력이 드랍되고 0.5s 를 넘으면 프로세스가 죽는다.
         sink 가 던진 예외도 frontend 치명으로 취급한다.
    """
    node = Node('frontend')
    for kind, path in (('seg', cfg.seg_engine), ('clip', cfg.clip_engine)):
        node.get_logger().info(f'{kind} engine = {path} ({os.path.getsize(path) / 2**20:.1f} MB)')
    from meridian_frontend.preprocessing.preprocessing import Preprocessing
    pre = Preprocessing(node, color_topic=cfg.rgb_topic, depth_topic=cfg.depth_topic)
    from meridian_frontend.pose.pose import Pose
    pose = Pose(node, pose_topic=cfg.pose_topic)
    from meridian_frontend.sam.sam import Sam
    sam = Sam(node, pre, pose, engine=cfg.seg_engine, info_topic=cfg.info_topic)
    from meridian_frontend.clip.clip import Clip
    clip = Clip(node, pre, sam, engine=cfg.clip_engine)
    from meridian_frontend.tracker.tracker import Tracker
    tracker = Tracker(node, pre, sam, clip=clip, info_topic=cfg.info_topic, nov_tau=cfg.nov_tau)
    from meridian_frontend.publisher.publisher import Publisher
    publisher = Publisher(node, sam, tracker, clip=clip, topic=cfg.tracklet_topic, sink=da_sink)

    pre.sink = sam.step                                       # 프레임 진행 = sink 직접 호출

    def sink(seq, stamp):
        tracker.step(seq, stamp)                              # gate 판정·clip 발화는 tracker 소유
        publisher.step(seq, stamp)

    sam.sink = sink
    nodes = [node]
    return nodes, dict(node=node, pre=pre, pose=pose, sam=sam, tracker=tracker, clip=clip,
                       publisher=publisher)


def main(argv=None):
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    rclpy.init()
    nodes, ctx = assemble(cfg)
    pre = ctx['pre']
    ex = SingleThreadedExecutor()
    for n in nodes:
        ex.add_node(n)
    print('[frontend] 조립 완료 — 입력 대기', flush=True)
    t0 = time.perf_counter()
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):    # Ctrl-C · launch 의 종료 신호(context 를 먼저 닫음)
        pass
    finally:
        dt = time.perf_counter() - t0
        n = pre.seq
        print(f'[frontend] 종료 — 프레임 {n} / {dt:.1f}s = {n / max(dt, 1e-9):.2f} Hz', flush=True)
        rclpy.try_shutdown()                                   # launch 경유면 context 가 이미 닫혀 있음 — 이중 shutdown 은 RCLError


if __name__ == '__main__':
    main()
