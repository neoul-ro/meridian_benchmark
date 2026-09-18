#!/usr/bin/env python3
"""uHumans2 언팩본을 meridian_frontend 에 프레임 단위로 넣고, publish(발행)되는 TrackletSet 을 전부 h5 로 저장한다.

왜 ROS 재생이 아니라 in-process 인가
  frontend 입력 구독은 BEST_EFFORT depth=1 이라 재생 속도가 처리 속도를 넘으면 프레임을 버린다.
  평가는 "GT 가 있으니 틀리면 안 된다" — 드랍 여부가 결과를 흔들면 안 되므로, frontend.py 가 실험 러너용으로
  열어 둔 assemble(cfg, da_sink) 로 조립하고 콜백(on_info/on_pose/on_pair)을 순서대로 직접 호출한다.
  프레임마다 publisher 방출 작업이 끝날 때까지 기다리므로 드랍 0 · 결정적이다. frontend 코드는 수정하지 않는다.

입력 계약 맞추기 (frontend 는 640x480 만 받는다)
  uHumans2 left_cam = 720x480, fx=fy=415.692, cx=360, cy=240
  → 가운데 640 폭을 잘라낸다 (좌우 40px). 스케일 변화 없음, cx 만 360→320.
    (frontend 기본 intrinsics 가 정확히 [415.692, 415.692, 320, 240] — 같은 방식으로 맞춰 개발된 값)
  rgb = rgb8, depth = 16UC1 mm (언팩본 PNG 그대로), pose = world_T_left_cam (optical, odom 을 이미지 stamp 로 보간)
  pose 는 이미지와 같은 stamp 로 넣는다 — frontend pose_at 은 "stamp 이하 최신 1개, 보간 없음".

출력 h5 (--out/frontend_output.h5)
  frames/{frame_idx, stamp_ns, n_det, kf_reason}      [F]  kf_reason: 0 없음 1 init(첫 keyframe) 2 cov(새 시야 비율 novelty 가
                                                            임계 초과) 3 birth(아직 임베딩 안 된 tracklet 이 생김) — tracker.py _decide
  kf/{seq, frame_idx, stamp_ns, n_obs, n_enc, obs_start}  [K]  publish 1건 = keyframe 1개 (검출 0개 keyframe 은 publish 안 함)
  obs/{kf_index, tracklet_id, points_num, points_start, depth_valid_ratio, seg_conf}  [O]  관측 1개 = 예측 검출(predicted detection), tracklet_id = 예측 ID(predicted ID)
  obs/{id_emb, query_emb}  float16 [O,512]  (--no-emb 이면 생략)
  obs/mask_bits  uint8 [O, 6144]  SAM 마스크 (192x256 격자, np.packbits). 격자 1칸 = 640x480 원본 2.5x2.5px
                 행 r·열 c 칸 = 원본 [2.5r, 2.5r+2.5) x [2.5c, 2.5c+2.5). TrackletSet 에는 없는 값 — 2D 분할 채점용
  points  float32 [P,3]  world(map) 좌표
  attrs: meta_json (인자·엔진 sha1·intrinsics·버전·요약)
"""
import os
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')   # frontend.py 와 같은 이유 (import torch 전에)

import argparse
import array
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS, MODELS  # noqa: E402  (meridian_benchmark 경로도 여기서 잡힘)
from meridian_benchmark.uhumans2 import UH2Sequence  # noqa: E402

W_IN, W_OUT, H = 720, 640, 480
CROP = (W_IN - W_OUT) // 2
KF_CODE = {'': 0, 'init': 1, 'cov': 2, 'birth': 3}


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def load_frame(seq, i):
    rgb = cv2.imread(str(seq.root / seq.rgb_files[i]), cv2.IMREAD_COLOR)          # BGR
    dep = cv2.imread(str(seq.root / seq.depth_files[i]), cv2.IMREAD_UNCHANGED)     # uint16 mm
    if rgb is None or dep is None:
        raise IOError(f'frame {i} 읽기 실패')
    if rgb.shape != (H, W_IN, 3) or dep.shape != (H, W_IN) or dep.dtype != np.uint16:
        raise ValueError(f'frame {i}: 예상과 다른 형식 rgb{rgb.shape} depth{dep.shape} {dep.dtype}')
    rgb = np.ascontiguousarray(rgb[:, CROP:CROP + W_OUT, ::-1])                   # → RGB, 640 폭
    dep = np.ascontiguousarray(dep[:, CROP:CROP + W_OUT])
    return rgb, dep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True, help='datasets/unpacked/uHumans2_<시퀀스>')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seg-engine', default=str(MODELS / 'frontend_rtx3060/fastsam.plan'))
    ap.add_argument('--clip-engine', default=str(MODELS / 'frontend_rtx3060/clip_image.plan'))
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None, help='마지막 프레임 번호 + 1')
    ap.add_argument('--no-emb', action='store_true', help='embedding 저장 생략 (파일 크기)')
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / 'frontend_output.h5'
    if h5_path.exists():
        raise SystemExit(f'이미 결과가 있음: {h5_path} — 섞이지 않게 다른 --out 을 쓸 것')

    seq = UH2Sequence(a.seq)
    ci = seq.camera_info('left_cam')
    K = ci['K']
    if (ci['width'], ci['height']) != (W_IN, H):
        raise SystemExit(f'예상 해상도 720x480 이 아님: {ci["width"]}x{ci["height"]}')
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] - CROP, K[1, 2]
    end = seq.n_frames if a.end is None else min(a.end, seq.n_frames)
    frames = np.arange(a.start, end)
    t_wc, q_wc = seq.camera_poses(seq.stamps_ns[frames], cam='left_cam')

    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    import torch
    import tensorrt as trt
    from meridian_frontend.frontend import assemble, parse_args

    captured = []
    cap_lock = threading.Lock()

    def sink(p):                                               # publisher-worker 스레드 · view 는 복사해서 보관
        rec = dict(seq=int(p.seq), stamp_ns=int(p.stamp.sec) * 1_000_000_000 + int(p.stamp.nanosec),
                   tracklet_ids=np.array(p.tracklet_ids, np.int64), points_num=np.array(p.points_num, np.int64),
                   points=np.array(p.points, np.float32), depth_valid_ratio=np.array(p.depth_valid_ratio, np.float32),
                   seg_conf=np.array(p.seg_conf, np.float32), n_enc=int(p.n_enc))
        if not a.no_emb:
            rec['id_emb'] = p.id_emb.astype(np.float16)
            rec['query_emb'] = p.query_emb.astype(np.float16)
        with cap_lock:
            captured.append(rec)

    rclpy.init(args=[])
    cfg = parse_args(['--seg-engine', a.seg_engine, '--clip-engine', a.clip_engine, '--tracklet-topic', ''])
    nodes, ctx = assemble(cfg, da_sink=sink)
    node, pre, pose, sam, tracker, clip, publisher = (ctx[k] for k in
                                                       ('node', 'pre', 'pose', 'sam', 'tracker', 'clip', 'publisher'))

    def stamp_msg(ns):
        return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))

    info = CameraInfo()
    info.width, info.height = W_OUT, H
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.header.frame_id = 'left_cam'
    sam.on_info(info)
    tracker.on_info(info)

    log_f = np.zeros(len(frames), np.int64)
    log_ndet = np.zeros(len(frames), np.int32)
    log_kf = np.zeros(len(frames), np.int8)
    seq_to_frame = {}
    masks_by_seq = {}                                          # keyframe seq → (N, 6144) packbits
    t_start = time.time()
    t_frame = []
    pool = ThreadPoolExecutor(max_workers=4)
    LOOKAHEAD = 16
    futs = {}
    try:
        for j, fi in enumerate(frames):
            for k in range(j, min(j + LOOKAHEAD, len(frames))):   # 디코딩 선행 (순서는 j 로 고정)
                if k not in futs:
                    futs[k] = pool.submit(load_frame, seq, int(frames[k]))
            rgb, dep = futs.pop(j).result()
            ns = int(seq.stamps_ns[fi])
            st = stamp_msg(ns)

            pm = PoseStamped()
            pm.header.stamp = st
            pm.header.frame_id = 'world'
            pm.pose.position.x, pm.pose.position.y, pm.pose.position.z = (float(v) for v in t_wc[j])
            (pm.pose.orientation.x, pm.pose.orientation.y,
             pm.pose.orientation.z, pm.pose.orientation.w) = (float(v) for v in q_wc[j])
            pose.on_pose(pm)

            cm = Image(); cm.header.stamp = st; cm.header.frame_id = 'left_cam'
            cm.height, cm.width, cm.encoding, cm.step = H, W_OUT, 'rgb8', W_OUT * 3
            cm.data = array.array('B', rgb.tobytes())
            dm = Image(); dm.header.stamp = st; dm.header.frame_id = 'left_cam'
            dm.height, dm.width, dm.encoding, dm.step = H, W_OUT, '16UC1', W_OUT * 2
            dm.data = array.array('B', dep.tobytes())

            t0 = time.perf_counter()
            seq_before = pre.seq
            pre.on_pair(cm, dm)                                 # → sam.step → tracker.step → publisher.step
            if pre.seq != seq_before + 1:
                raise RuntimeError(f'frame {fi}: frontend 가 프레임을 받지 않음 (입력 계약 로그 확인)')
            if sam.n_nopose:
                raise RuntimeError(f'frame {fi}: pose 미준비로 frontend 가 프레임을 건너뜀')
            if not publisher.job_done.wait(timeout=10.0):       # 방출 끝날 때까지 대기 → 드랍 0
                raise RuntimeError(f'frame {fi}: publisher 방출 10s 초과')
            publisher.check_alive(); clip.check_alive()
            t_frame.append(time.perf_counter() - t0)

            if tracker.is_kf and sam.det.numel() > 0:          # 발행 관측과 같은 순서(det 순)의 마스크
                mk = sam.masks.index_select(0, sam.det).reshape(int(sam.det.numel()), -1).cpu().numpy()
                masks_by_seq[pre.seq] = np.packbits(mk, axis=1)
            log_f[j] = fi
            log_ndet[j] = int(sam.det.numel())
            log_kf[j] = KF_CODE[tracker.kf_reason] if tracker.is_kf else 0
            seq_to_frame[pre.seq] = int(fi)
            if (j + 1) % 500 == 0 or j + 1 == len(frames):
                el = time.time() - t_start
                print(f'[run] {j + 1}/{len(frames)} 프레임 · kf {int((log_kf[:j + 1] > 0).sum())} · '
                      f'발행 {len(captured)} · {(j + 1) / el:.1f} fps · frontend {np.mean(t_frame[-500:]) * 1e3:.1f} ms/프레임',
                      flush=True)
        publisher.job_done.wait(timeout=10.0)
        clip.enq_done.wait(timeout=10.0)
        torch.cuda.synchronize()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    summary = tracker.summary() + publisher.summary()
    for line in summary:
        print(line, flush=True)

    # ---- 저장 ----
    captured.sort(key=lambda r: r['seq'])
    n_kf_logged = int((log_kf > 0).sum())
    kf_with_det = sum(1 for j in range(len(frames)) if log_kf[j] > 0 and log_ndet[j] > 0)
    if len(captured) != kf_with_det:                           # publisher 는 검출 0 인 kf 는 방출하지 않는다
        raise RuntimeError(f'발행 수 불일치: 캡처 {len(captured)} vs 검출 있는 kf {kf_with_det}')
    O = sum(len(r['tracklet_ids']) for r in captured)
    P = sum(len(r['points']) for r in captured)
    with h5py.File(h5_path, 'w') as f:
        g = f.create_group('frames')
        g['frame_idx'] = log_f; g['stamp_ns'] = seq.stamps_ns[log_f]; g['n_det'] = log_ndet; g['kf_reason'] = log_kf
        g = f.create_group('kf')
        g['seq'] = np.array([r['seq'] for r in captured], np.int64)
        g['frame_idx'] = np.array([seq_to_frame[r['seq']] for r in captured], np.int64)
        g['stamp_ns'] = np.array([r['stamp_ns'] for r in captured], np.int64)
        g['n_obs'] = np.array([len(r['tracklet_ids']) for r in captured], np.int64)
        g['n_enc'] = np.array([r['n_enc'] for r in captured], np.int64)
        g['obs_start'] = np.concatenate(([0], np.cumsum(g['n_obs'][:])[:-1])).astype(np.int64) if captured else np.zeros(0, np.int64)
        if captured and not np.array_equal(g['stamp_ns'][:], seq.stamps_ns[g['frame_idx'][:]]):
            raise RuntimeError('kf stamp ↔ frame 번호 대응 불일치')
        g = f.create_group('obs')
        kf_index = np.concatenate([np.full(len(r['tracklet_ids']), k, np.int64) for k, r in enumerate(captured)]) if O else np.zeros(0, np.int64)
        g['kf_index'] = kf_index
        g['tracklet_id'] = np.concatenate([r['tracklet_ids'] for r in captured]) if O else np.zeros(0, np.int64)
        pn = np.concatenate([r['points_num'] for r in captured]) if O else np.zeros(0, np.int64)
        g['points_num'] = pn
        g['points_start'] = np.concatenate(([0], np.cumsum(pn)[:-1])).astype(np.int64) if O else np.zeros(0, np.int64)
        g['depth_valid_ratio'] = np.concatenate([r['depth_valid_ratio'] for r in captured]) if O else np.zeros(0, np.float32)
        g['seg_conf'] = np.concatenate([r['seg_conf'] for r in captured]) if O else np.zeros(0, np.float32)
        if not a.no_emb and O:
            g.create_dataset('id_emb', data=np.concatenate([r['id_emb'] for r in captured]), compression='gzip', compression_opts=1)
            g.create_dataset('query_emb', data=np.concatenate([r['query_emb'] for r in captured]), compression='gzip', compression_opts=1)
        if O:
            mb = []
            for r in captured:
                m = masks_by_seq.get(r['seq'])
                if m is None or len(m) != len(r['tracklet_ids']):
                    raise RuntimeError(f"seq={r['seq']} 마스크 수 불일치: {None if m is None else len(m)} vs 관측 {len(r['tracklet_ids'])}")
                mb.append(m)
            md = g.create_dataset('mask_bits', data=np.concatenate(mb), compression='gzip', compression_opts=4, chunks=True)
            md.attrs['grid_hw'] = (sam.PH, sam.PW); md.attrs['px_per_cell'] = 2.5
        pts = np.concatenate([r['points'] for r in captured]) if P else np.zeros((0, 3), np.float32)
        if int(pn.sum()) != len(pts):
            raise RuntimeError(f'points_num 합 {int(pn.sum())} != 점 수 {len(pts)}')
        f.create_dataset('points', data=pts, compression='gzip', compression_opts=1, chunks=True)
        meta = dict(
            sequence=seq.root.name, dataset=str(seq.root), frames=[int(a.start), int(end)],
            n_frames=int(len(frames)), crop_px=CROP, intrinsics_640=[fx, fy, cx, cy],
            pose='world_T_left_cam (optical), odom slerp/linear → image stamp', depth='16UC1 mm (unpacked PNG)',
            feed='in-process assemble(cfg, da_sink), 프레임마다 방출 완료 대기 (드랍 0)',
            seg_engine=a.seg_engine, seg_engine_sha1=sha1(a.seg_engine),
            clip_engine=a.clip_engine, clip_engine_sha1=sha1(a.clip_engine),
            tensorrt=trt.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
            n_kf=n_kf_logged, n_kf_published=len(captured), n_obs=int(O), n_points=int(P),
            n_tracklet_ids=int(len(np.unique(f['obs/tracklet_id'][:]))) if O else 0,
            frontend_ms_mean=float(np.mean(t_frame) * 1e3), frontend_ms_p95=float(np.percentile(t_frame, 95) * 1e3),
            wall_s=float(time.time() - t_start), summary=summary,
            built_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        f.attrs['meta_json'] = json.dumps(meta, ensure_ascii=False)
    (out / 'run_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f'[run] 저장 {h5_path} — kf {len(captured)} · 관측 {O} · 점 {P} · tracklet id {meta["n_tracklet_ids"]}', flush=True)
    node.destroy_node()
    rclpy.try_shutdown()
    os._exit(0)                                                 # daemon 워커 스레드(clip/publisher)가 종료를 막지 않게


if __name__ == '__main__':
    main()
