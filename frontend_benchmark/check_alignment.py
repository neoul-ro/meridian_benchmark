#!/usr/bin/env python3
"""입력 주입이 맞았는지 검증 — 채점 전에 반드시 통과해야 한다 (eval.sh 정렬 게이트).

(1) 입력 정렬: frontend 가 publish(발행)한 world 점 ↔ 같은 프레임 depth 를 이 스크립트가 독립적으로 역투영한 점
    (crop 640, cx=320, world_T_left_cam). crop·intrinsics·pose 규약 중 하나라도 틀리면 수십 cm~m 로 벌어진다.
    게이트: p99 ≤ INPUT_P99_CM(3cm)
(2) 프레임 번호 밀림 (감사 C17): 프레임 번호가 통째로 한 칸 밀리면 (depth 와 pose 가 함께 밀림) 정적 장면에서는 이웃 프레임도
    같은 표면이라 (1) 의 분포가 거의 그대로다 (감사 a02: ±2 칸 밀어도 p50 0.33cm · p90 0.79cm 로 통과).
    그런데 frontend 점은 proto 격자 픽셀의 depth 를 그대로 역투영한 점이라, 맞는 프레임의 조밀 역투영에는 점이 '정확히' 겹친다.
    그래서 keyframe 마다 주장한 프레임 f 의 이웃 c ∈ {−1, 0, +1} 로 depth(f+c)+pose(f+c) 조밀 역투영을 만들고
      within_1mm(c) = frontend 점(3개 중 1개 표본) 중 1mm 이내 비율
    을 비교한다. 게이트: within_1mm(0) ≥ SHIFT_MIN_WITHIN(0.10) 이고 within_1mm(0) ≥ SHIFT_RATIO(2)·max(within_1mm(±1)).
    실측 (감사 fix_C c17_shift_probe.json, keyframe 30개): 맞는 프레임 0.25 · 한 칸 밀린 프레임 0.02~0.06.
    ±1 칸 밀림은 비율 조건으로, ±2 칸 이상은 절대 조건으로 걸린다.
(3) GT 정렬(참고, 게이트 아님): frontend 점 ↔ 그 프레임에 활성인 GT 트랙(ground-truth track) 점(2cm 복셀).
    물체 위 점이면 수 cm 이내 (벽·바닥처럼 GT 에서 제외된 표면 위 점은 멀다 — 그래서 분포 하위 분위수를 본다).
    정적 GT 점은 프레임과 무관한 world 좌표라 프레임 밀림은 여기서 안 보인다.

출력 <--out 또는 --run>/alignment_check.json
  input_alignment_cm · input_within_3cm · gt_alignment_cm · gt_within_5cm · gt_within_20cm   (기존 키)
  frame_shift {"-1"|"0"|"1": {within_1mm, p25_cm, n_points}} · gate {input_p99_ok, frame_shift_ok, pass, 기준값}
  run_h5 {path, size, mtime_ns, sha1}   ← eval.sh 게이트가 '지금 h5 를 검증한 결과' 인지 확인하는 데 쓴다 (감사 C7)
종료 코드 0 = 게이트 통과.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS, MODELS  # noqa: E402,F401  (meridian_benchmark 경로도 여기서 잡힘)
from meridian_benchmark.uhumans2 import UH2Sequence  # noqa: E402

CROP = 40
INPUT_P99_CM = 3.0
EXACT_M = 0.001
SHIFT_MIN_WITHIN = 0.10
SHIFT_RATIO = 2.0
SHIFT_SUB = 3


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--seq', required=True)
    ap.add_argument('--gt', required=True, help='meridian_benchmark/tracklets/<seq>.h5')
    ap.add_argument('--n-kf', type=int, default=20)
    ap.add_argument('--out', default=None, help='alignment_check.json 을 쓸 폴더 (기본 --run)')
    a = ap.parse_args()

    seq = UH2Sequence(a.seq)
    K = seq.camera_info()['K'].copy()
    K[0, 2] -= CROP
    h5_path = Path(a.run) / 'frontend_output.h5'
    f = h5py.File(h5_path, 'r')
    kf_frame = f['kf/frame_idx'][:]; obs_start = f['kf/obs_start'][:]; n_obs = f['kf/n_obs'][:]
    pstart = f['obs/points_start'][:]; pnum = f['obs/points_num'][:]
    P = f['points']
    g = h5py.File(a.gt, 'r')
    gi = {k: g['index'][k][:] for k in ('tracklet_id', 'first_frame', 'last_frame')}
    vv, uu = np.mgrid[0:480, 0:640]
    trees = {}

    def dense(fi):
        """depth(fi) + pose(fi) 조밀 역투영 KD-tree (0.1 < z < 5m)."""
        if fi not in trees:
            if len(trees) > 8:
                trees.clear()
            dep = cv2.imread(str(seq.root / seq.depth_files[fi]), cv2.IMREAD_UNCHANGED)[:, CROP:CROP + 640].astype(np.float32) / 1000.0
            z = dep; ok = (z > 0.1) & (z < 5.0)
            cam = np.stack([(uu[ok] - K[0, 2]) / K[0, 0] * z[ok], (vv[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]], 1)
            t, q = seq.camera_poses(seq.stamps_ns[[fi]], cam='left_cam')
            trees[fi] = cKDTree(cam @ Rotation.from_quat(q[0]).as_matrix().T + t[0])
        return trees[fi]

    pick = np.linspace(0, len(kf_frame) - 1, min(a.n_kf, len(kf_frame))).astype(int)
    d_in, d_gt = [], []
    shift = {c: [] for c in (-1, 0, 1)}
    for k in pick:
        fi = int(kf_frame[k])
        o0, o1 = obs_start[k], obs_start[k] + n_obs[k]
        if o1 <= o0:
            continue
        s, e = pstart[o0], pstart[o1 - 1] + pnum[o1 - 1]
        pts = P[s:e]
        if len(pts) == 0 or not (0 <= fi < seq.n_frames):
            continue
        d_in.append(dense(fi).query(pts, k=1)[0])
        if 1 <= fi < seq.n_frames - 1:
            sub = pts[::SHIFT_SUB]
            for c in (-1, 0, 1):
                shift[c].append(dense(fi + c).query(sub, k=1)[0])
        act = np.flatnonzero((gi['first_frame'] <= fi) & (gi['last_frame'] >= fi))
        if len(act):
            gp = np.concatenate([g[f'tracklets/{int(gi["tracklet_id"][i]):05d}/tracklet_geometry/points'][:] for i in act])
            d_gt.append(cKDTree(gp).query(pts, k=1)[0])
    d_in = np.concatenate(d_in) if d_in else np.zeros(0); d_gt = np.concatenate(d_gt) if d_gt else np.zeros(0)
    q = lambda d: {f'p{p}': round(float(np.percentile(d, p)) * 100, 2) for p in (10, 25, 50, 90, 99)}
    fs = {}
    for c, arrs in shift.items():
        dd = np.concatenate(arrs) if arrs else np.zeros(0)
        fs[str(c)] = dict(within_1mm=round(float((dd <= EXACT_M).mean()), 4) if len(dd) else None,
                          p25_cm=round(float(np.percentile(dd, 25)) * 100, 3) if len(dd) else None, n_points=int(len(dd)))
    w0, wm, wp = fs['0']['within_1mm'], fs['-1']['within_1mm'], fs['1']['within_1mm']
    shift_ok = bool(w0 is not None and wm is not None and wp is not None and w0 >= SHIFT_MIN_WITHIN and w0 >= SHIFT_RATIO * max(wm, wp))
    input_ok = bool(len(d_in) and q(d_in)['p99'] <= INPUT_P99_CM)
    st = os.stat(h5_path)
    rep = {'n_kf_checked': int(len(pick)), 'n_points': int(len(d_in)),
           'input_alignment_cm': q(d_in) if len(d_in) else None,
           'input_within_3cm': round(float((d_in <= 0.03).mean()), 4) if len(d_in) else None,
           'gt_alignment_cm': q(d_gt) if len(d_gt) else None,
           'gt_within_5cm': round(float((d_gt <= 0.05).mean()), 4) if len(d_gt) else None,
           'gt_within_20cm': round(float((d_gt <= 0.20).mean()), 4) if len(d_gt) else None,
           'frame_shift': fs,
           'gate': {'input_p99_ok': input_ok, 'frame_shift_ok': shift_ok, 'pass': bool(input_ok and shift_ok),
                    'input_p99_cm_max': INPUT_P99_CM, 'exact_m': EXACT_M, 'shift_min_within': SHIFT_MIN_WITHIN, 'shift_ratio': SHIFT_RATIO},
           'run_h5': dict(path=str(h5_path.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns, sha1=sha1(h5_path))}
    ok = rep['gate']['pass']
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print('입력 정렬:', 'PASS' if input_ok else f'FAIL (p99 > {INPUT_P99_CM}cm)')
    print('프레임 밀림:', 'PASS' if shift_ok else
          f'FAIL (1mm 이내 비율 −1/0/+1 = {wm} / {w0} / {wp}; 0 이 {SHIFT_MIN_WITHIN} 이상이고 이웃의 {SHIFT_RATIO}배 이상이어야 함)')
    out = Path(a.out) if a.out else Path(a.run)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'alignment_check.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
