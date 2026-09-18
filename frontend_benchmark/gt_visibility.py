#!/usr/bin/env python3
"""GT 트랙(ground-truth track, 코드·CSV 의 에피소드)이 프레임마다 "frontend 가 볼 수 있는 영역"에 몇 픽셀 가시(visible)였는지 계산한다 (시퀀스당 1회, 캐시).

왜 필요한가
  GT 에피소드는 원본 720 폭·모든 거리 기준으로 정의됐다. frontend 에는 가운데 640 폭만 넣었고 frontend 는
  z<5m 만 쓴다. 잘린 좌우 40px 에만 보였거나 5m 밖에서만 보인 물체를 "놓침"으로 세면 frontend 탓이 아닌데
  점수가 깎인다. 또 frontend 는 원본 40x40px(=1600px) 미만 마스크를 버리도록 설계됐다(sam.AREA_MIN).
  → 프레임별 가시 픽셀 수가 있어야 "볼 수 있었는데 놓친 것"과 "애초에 못 보는 것"을 나눌 수 있다.
  채점 대상(eligible) = GT 트랙 동안 한 프레임이라도 px_crop ≥ 1600 · 평가 거리 범위(evaluation range) = 카메라 5m.
  2D 라벨(gt_labels_2d)의 채점 대상(라벨 면적 ≥ 256칸)과는 정의가 다르다 — README '가시·채점 대상 두 정의' 참고.

방법
  seg 색 = prefab(재질) 타입이지 인스턴스가 아니다(같은 색 물체 여러 개). 그래서
  색 마스크 픽셀을 depth 로 역투영(stride 2) → 카메라 5m 이내 → 그 GT 트랙 점군(2cm 복셀)에서 tol(6cm) 이내
  인 픽셀만 그 GT 트랙 것으로 센다. (GT 점군은 stride 4 역투영의 2cm 복셀이라 5m 에서 표본 간격 ~4.8cm → 6cm)
  px = 표본 수 × stride² (원본 해상도 픽셀로 환산).

출력 npz: ep_index, frame, px_full(720 폭 전체), px_crop(가운데 640 폭) — 0 인 쌍은 저장 안 함.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS, MODELS  # noqa: E402  (meridian_benchmark 경로도 여기서 잡힘)
from meridian_benchmark.uhumans2 import UH2Sequence, unproject  # noqa: E402

CROP_LO, CROP_HI = 40, 680
G = {}


def _init(seq_dir, gt_path, stride, max_range, tol):
    G['seq'] = UH2Sequence(seq_dir)
    f = h5py.File(gt_path, 'r')
    ix = f['index']
    G['tid'] = ix['tracklet_id'][:]; G['first'] = ix['first_frame'][:]; G['last'] = ix['last_frame'][:]
    G['color'] = np.array([int((lambda c: c.decode() if isinstance(c, bytes) else c)(
        f[f'tracklets/{int(t):05d}/_metadata/color_hex'][()]).lstrip('#'), 16) for t in G['tid']], np.int64)
    G['h5'] = f
    G['trees'] = {}
    G['K'] = G['seq'].camera_info()['K']
    G['stride'], G['max_range'], G['tol'] = stride, max_range, tol


def _tree(i):
    if i not in G['trees']:
        p = G['h5'][f'tracklets/{int(G["tid"][i]):05d}/tracklet_geometry/points'][:]
        G['trees'][i] = cKDTree(p)
    return G['trees'][i]


def _frame(fi):
    seq = G['seq']
    act = np.flatnonzero((G['first'] <= fi) & (G['last'] >= fi))
    out = []
    if not len(act):
        return out
    seg = seq.seg_packed(fi)
    dep = seq.depth(fi)
    t, q = seq.camera_poses(seq.stamps_ns[[fi]], cam='left_cam')
    R = Rotation.from_quat(q[0]).as_matrix()
    s2 = G['stride'] ** 2
    for c in np.unique(G['color'][act]):
        m = seg == c
        if not m.any():
            continue
        pts, (vv, uu) = unproject(dep, G['K'], stride=G['stride'], valid_mask=m)
        if not len(pts):
            continue
        near = np.linalg.norm(pts, axis=1) <= G['max_range']
        pts, uu = pts[near], uu[near]
        if not len(pts):
            continue
        W = pts @ R.T + t[0]
        incrop = (uu >= CROP_LO) & (uu < CROP_HI)
        for i in act[G['color'][act] == c]:
            d = _tree(i).query(W, k=1, distance_upper_bound=G['tol'])[0]
            close = np.isfinite(d)
            nf = int(close.sum())
            if nf:
                out.append((int(i), int(fi), nf * s2, int((close & incrop).sum()) * s2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True); ap.add_argument('--gt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=2); ap.add_argument('--max-range', type=float, default=5.0)
    ap.add_argument('--tol', type=float, default=0.06); ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    n = UH2Sequence(a.seq).n_frames
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(a.seq, a.gt, a.stride, a.max_range, a.tol)) as pool:
        for k, r in enumerate(pool.imap(_frame, range(n), chunksize=16)):
            rows.extend(r)
            if (k + 1) % 2000 == 0:
                print(f'[vis] {k + 1}/{n} 프레임 · {time.time() - t0:.0f}s', flush=True)
    arr = np.array(rows, np.int64).reshape(-1, 4)
    np.savez_compressed(a.out, ep_index=arr[:, 0], frame=arr[:, 1], px_full=arr[:, 2], px_crop=arr[:, 3],
                        meta=json.dumps(dict(seq=a.seq, gt=a.gt, stride=a.stride, max_range=a.max_range, tol=a.tol,
                                             crop=[CROP_LO, CROP_HI], n_frames=n, seconds=round(time.time() - t0, 1))))
    print(f'[vis] 저장 {a.out} — 쌍 {len(arr)} · {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
