#!/usr/bin/env python3
"""keyframe GT 인스턴스의 난이도 — 잘림(truncation) · 가림 비율(occlusion ratio) · 최소 bbox 변 → Easy/Moderate/Hard (KITTI 기준 변형).

왜
  "탐지 가능" 을 면적(1600px) 하나로 자르면 화면 끝에 폭 12px 로 걸린 띠까지 채점 대상이 된다(뷰어로 확인).
  KITTI 는 잘림·가림·크기로 쉬움/보통/어려움을 나눠 따로 채점한다. 같은 방식을 쓴다.
  표준 = KITTI object devkit (사본 _audit/B_2d_geometry/refs/kitti): readme.txt:60-61 "truncated refers to the object
  leaving image boundaries", cpp/evaluate_object.cpp:44-46 (MIN_HEIGHT {40,25,25} · MAX_OCCLUSION · MAX_TRUNCATION
  {0.15,0.3,0.5}), :413 무시 조건 "occlusion>MAX || truncation>MAX || height<=MIN_HEIGHT".

재는 법 (keyframe 프레임 f, GT 에피소드 e)
  실루엣   e 와 같은 gt_object_id 의 '모든 에피소드' GT 점군 합집합(2cm 복셀, 최대 30000점 표본)을 f 의 카메라로 투영해
           점마다 반지름 r = ceil(0.5 · 0.02 · fx / z) 픽셀로 칠한 영역. 640x480 밖까지 그린다(잘림 계산용).
           → "가려지지 않았다면 보였을 물체 전체 윤곽". 에피소드 점군은 그 연속 구간에 본 면뿐이라, 화면 가장자리에
           걸쳐서만 보였던 물체는 안 보인 부분이 빠져 잘림이 과소평가된다 — 그래서 물체 전체(합집합)를 쓴다 (KITTI 는
           물체 모델 전체를 역투영해 잘림을 잰다, readme.txt:23-25). 뒷면 점이 투영돼도 앞면 윤곽 안에 들어가므로
           자기 가림은 영향 없다.
  잘림     1 − (화면 안 실루엣) / (실루엣 전체). 실루엣 전체에는 카메라에서 볼 수 없는 점도 '화면 밖' 으로 넣는다
             near plane(z ≤ 0.1m) 뒤 점, 640x480 을 가운데 둔 3배 캔버스 밖으로 투영되는 점
           이 점들은 그릴 수 없으니 점마다 상 면적 (0.02·fx / max(z, 0.1))² (near plane 에서 자른 깊이)을 더하고,
           그린 점들의 (칠한 합집합 픽셀 / 상 면적 합) 비율 ρ 로 보정한다 (겹침·칠하기 반지름 효과를 같은 물체에서 맞춤).
           예전에는 이 점들을 분모에서 버려 잘림이 과소평가됐다 (감사 B 결함 7).
  가림     점 단위 depth 검사 (에피소드 점군 — 이 구간에 본 면). 화면 안에 떨어진 GT 점마다 그 픽셀이
             같은 인스턴스 라벨이면 → 보임
             다른 표면이고 측정 depth 가 점보다 tol(max(5cm, 3%·z)) 이상 앞이면 → 가려짐
             그 밖(뒤쪽 표면·depth 없음·반사 등)은 판단에서 뺀다
           가림 = 가려짐 / (보임 + 가려짐). 실루엣 면적 비는 얇은 물체에서 부풀고(점 칠하기 반지름),
           유리 반사 인스턴스에서 어긋나 쓰지 않는다(그림으로 확인).
  최소 bbox 변  실제 보인 픽셀 bbox 의 min(w, h) — KITTI 의 'Min. bounding box height' 를 min(폭, 높이)로 바꾼 것
  테두리   라벨이 640x480 테두리(첫·끝 행/열)에 닿는가
  사람(움직이는 물체)은 에피소드 점군이 이동 경로에 퍼져 실루엣이 의미 없다 → 잘림·가림 없음(-1).
  대신 라벨이 테두리에 닿으면 잘렸을 수 있으므로 easy 에서 뺀다 (감사 B 결함 13).

등급 Easy / Moderate / Hard (KITTI object 기준을 물체 일반에 맞춘 변형(adapted): 높이 대신 최소 bbox 변, 경계는 evaluate_object.cpp:413 과 같게)
  KITTI 의 가림(occluded)은 0~3 범주값이다. 여기서는 연속값 가림 비율에 20% · 50% 경계를 우리가 정해 범주를 환산했다.
  easy      최소 폭 > 40px · 잘림 ≤ 15% · 가림 ≤ 20% · (잘림 모름(-1) 이면 테두리 안 닿음)
  moderate  최소 폭 > 25px · 잘림 ≤ 30% · 가림 ≤ 50%
  hard      최소 폭 > 25px · 잘림 ≤ 50%
  그 밖      beyond = 제외(ignored, 채점 등급 밖)
  KITTI 는 height <= MIN_HEIGHT 이면 무시 → 폭 40 은 easy 아님, 25 는 등급 밖 (예전 코드는 ≥ 로 포함했다, 결함 12).
  채점은 KITTI 처럼 누적: moderate 에는 easy 가, hard 에는 moderate 가 포함된다.

출력 <labels_dir>/difficulty.npz : frame, ep_index, trunc, occ, min_dim, sil_px, vis_px, level(0 easy 1 moderate 2 hard 3 beyond)
      + at_border (라벨이 테두리에 닿음) · sil_total_px (화면 밖 포함 실루엣 전체 상 면적)
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS  # noqa: E402,F401
from meridian_benchmark.uhumans2 import UH2Sequence  # noqa: E402

CROP = 40
HUMAN_COLORS = ('23d5ea',)
MAX_PTS = 30000
VOX = 0.02
NEAR = 0.1                 # near plane (m)
LEVELS = ('easy', 'moderate', 'hard', 'beyond')
G = {}


def level_of(min_dim, trunc, occ, at_border=False):
    """등급 번호. trunc/occ 가 음수(사람 — 모름)면 크기만으로 판정하되, 테두리에 닿으면 easy 제외.
    KITTI evaluate_object.cpp:413 — height<=MIN_HEIGHT 무시(→ 폭은 strict >), truncation>MAX · occlusion>MAX 무시(→ ≤)."""
    unknown = trunc < 0
    t = 0.0 if unknown else trunc
    o = 0.0 if occ < 0 else occ
    if min_dim > 40 and t <= 0.15 and o <= 0.20 and not (unknown and at_border):
        return 0
    if min_dim > 25 and t <= 0.30 and o <= 0.50:
        return 1
    if min_dim > 25 and t <= 0.50:
        return 2
    return 3


def _init(seq_dir, gt_path, labels_dir):
    G['seq'] = UH2Sequence(seq_dir)
    f = h5py.File(gt_path, 'r'); ix = f['index']
    G['h5'] = f; G['tid'] = ix['tracklet_id'][:]; G['oid'] = ix['gt_object_id'][:]
    G['human'] = np.array([str((lambda c: c.decode() if isinstance(c, bytes) else c)(
        f[f'tracklets/{int(t):05d}/_metadata/color_hex'][()])).lstrip('#') in HUMAN_COLORS for t in G['tid']])
    K = G['seq'].camera_info()['K'].copy(); K[0, 2] -= CROP
    G['K'] = K; G['labels'] = Path(labels_dir); G['pts'] = {}; G['obj_pts'] = {}


def _points(e):
    if e not in G['pts']:
        p = G['h5'][f'tracklets/{int(G["tid"][e]):05d}/tracklet_geometry/points'][:]
        if len(p) > MAX_PTS:
            p = p[np.random.default_rng(e).choice(len(p), MAX_PTS, replace=False)]
        if len(G['pts']) > 400:
            G['pts'].clear()
        G['pts'][e] = p.astype(np.float64)
    return G['pts'][e]


def _obj_points(e):
    """에피소드 e 와 같은 gt_object_id 의 모든 에피소드 점군 합집합 (2cm 복셀 중복 제거, 최대 MAX_PTS 표본)."""
    oid = int(G['oid'][e])
    if oid not in G['obj_pts']:
        eps = np.flatnonzero(G['oid'] == oid)
        p = np.concatenate([G['h5'][f'tracklets/{int(G["tid"][k]):05d}/tracklet_geometry/points'][:] for k in eps])
        _, first = np.unique(np.floor(p / VOX).astype(np.int64), axis=0, return_index=True)
        p = p[np.sort(first)]
        if len(p) > MAX_PTS:
            p = p[np.random.default_rng(oid).choice(len(p), MAX_PTS, replace=False)]
        if len(G['obj_pts']) > 200:
            G['obj_pts'].clear()
        G['obj_pts'][oid] = p.astype(np.float64)
    return G['obj_pts'][oid]


def silhouette(pts, R, t, K):
    """→ (화면 안 실루엣 픽셀 수, 실루엣 전체 픽셀 수 — 화면 밖 포함).
    near plane(z ≤ NEAR) 뒤 점과 3배 캔버스 밖 점은 그리지 않고 '화면 밖' 상 면적으로 더한다 (모듈 docstring '잘림')."""
    if not len(pts):
        return 0, 0
    cam = (pts - t) @ R
    z = cam[:, 2]
    foot = (VOX * K[0, 0] / np.maximum(z, NEAR)) ** 2           # 점 하나의 상 면적 (near plane 에서 자른 깊이)
    drawn = np.zeros(len(z), bool)
    front = z > NEAR
    if front.any():
        u = np.full(len(z), -1e9); v = np.full(len(z), -1e9)
        u[front] = K[0, 0] * cam[front, 0] / z[front] + K[0, 2]
        v[front] = K[1, 1] * cam[front, 1] / z[front] + K[1, 2]
        drawn = front & (u > -640) & (u < 1280) & (v > -480) & (v < 960)
    inside = canvas_px = 0
    rho = 1.0
    if drawn.any():
        zd = z[drawn]
        r = np.clip(np.ceil(0.5 * VOX * K[0, 0] / zd), 1, 12).astype(int)
        ui, vi = u[drawn].astype(int) + 640, v[drawn].astype(int) + 480
        x0, x1 = max(ui.min() - 13, 0), min(ui.max() + 14, 1920)
        y0, y1 = max(vi.min() - 13, 0), min(vi.max() + 14, 1440)
        canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for rad in np.unique(r):
            sel = r == rad
            layer = np.zeros_like(canvas)
            layer[vi[sel] - y0, ui[sel] - x0] = 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
            canvas |= cv2.dilate(layer, k)
        canvas_px = int(canvas.sum())
        ix0, ix1 = max(640 - x0, 0), max(min(1280 - x0, canvas.shape[1]), 0)
        iy0, iy1 = max(480 - y0, 0), max(min(960 - y0, canvas.shape[0]), 0)
        inside = int(canvas[iy0:iy1, ix0:ix1].sum()) if ix1 > ix0 and iy1 > iy0 else 0
        rho = canvas_px / max(float(foot[drawn].sum()), 1e-9)
    extra = rho * float(foot[~drawn].sum())
    return inside, int(round(canvas_px + extra))


def occlusion(pts, R, t, K, lab, dep, val):
    cam = (pts - t) @ R
    z = cam[:, 2]; ok = z > NEAR
    u = (K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]).astype(int); v = (K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]).astype(int); z = z[ok]
    inb = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    u, v, z = u[inb], v[inb], z[inb]
    if not len(z):
        return 0.0, 0
    d = dep[v, u]; same = lab[v, u] == val
    tol = np.maximum(0.05, 0.03 * z)
    visible = same & (d > 0)
    occluded = (~same) & (d > 0) & (d < z - tol)
    n = int(visible.sum() + occluded.sum())
    return (float(occluded.sum()) / n if n else 0.0), n


def frame_rows(fi):
    """→ [(frame, ep, trunc, occ, min_dim, sil_px, vis_px, level, at_border, sil_total_px), ...]"""
    seq, K = G['seq'], G['K']
    lab = cv2.imread(str(G['labels'] / f'{fi:06d}.png'), cv2.IMREAD_UNCHANGED)
    dep = seq.depth(fi)[:, CROP:CROP + 640]
    ids = np.unique(lab[lab > 0])
    if not len(ids):
        return []
    t, q = seq.camera_poses(seq.stamps_ns[[fi]], cam='left_cam')
    R = Rotation.from_quat(q[0]).as_matrix()
    out = []
    for val in ids:
        e = int(val) - 1
        ys, xs = np.nonzero(lab == val)
        vis = len(xs)
        min_dim = int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
        at_border = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == lab.shape[1] - 1 or ys.max() == lab.shape[0] - 1)
        if G['human'][e]:
            trunc = occ = -1.0; sil_in = sil_tot = -1
        else:
            sil_in, sil_tot = silhouette(_obj_points(e), R, t[0], K)
            trunc = 1.0 - sil_in / sil_tot if sil_tot else 0.0
            occ, _ = occlusion(_points(e), R, t[0], K, lab, dep, val)
        out.append((fi, e, trunc, occ, min_dim, sil_in, vis, level_of(min_dim, trunc, occ, at_border), int(at_border), sil_tot))
    return out


def build(seq_dir, gt_path, labels_dir, frames, workers=8, log=print):
    t0 = time.time()
    labels_dir = Path(labels_dir)
    frames = sorted(set(int(x) for x in frames))
    rows = []
    with Pool(workers, initializer=_init, initargs=(str(seq_dir), str(gt_path), str(labels_dir))) as pool:
        for r in pool.imap(frame_rows, frames, chunksize=4):
            rows.extend(r)
    a = np.array(rows, dtype=np.float64).reshape(-1, 10)
    np.savez_compressed(labels_dir / 'difficulty.npz', frame=a[:, 0].astype(np.int64), ep_index=a[:, 1].astype(np.int64),
                        trunc=a[:, 2].astype(np.float32), occ=a[:, 3].astype(np.float32), min_dim=a[:, 4].astype(np.int32),
                        sil_px=a[:, 5].astype(np.int64), vis_px=a[:, 6].astype(np.int64), level=a[:, 7].astype(np.int8),
                        at_border=a[:, 8].astype(bool), sil_total_px=a[:, 9].astype(np.int64),
                        meta=json.dumps(dict(levels=LEVELS, max_pts=MAX_PTS, splat='r=ceil(0.5*0.02*fx/z)',
                                             silhouette='같은 gt_object_id 전 에피소드 합집합',
                                             offscreen='near plane(z<=0.1) 뒤·3배 캔버스 밖 점 = 화면 밖 상 면적 (0.02 fx/max(z,0.1))^2 x rho',
                                             min_dim='strict > 40/25 (KITTI evaluate_object.cpp:413)',
                                             human='잘림·가림 -1, 테두리 닿으면 easy 제외'), ensure_ascii=False))
    log(f'[difficulty] {len(frames)} 프레임 · 인스턴스 {len(a)} · {time.time() - t0:.0f}s')
    return labels_dir / 'difficulty.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--seq', required=True)
    ap.add_argument('--gt', required=True); ap.add_argument('--labels', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    frames = h5py.File(Path(a.run) / 'frontend_output.h5', 'r')['kf/frame_idx'][:]
    build(a.seq, a.gt, a.labels, frames, a.workers)


if __name__ == '__main__':
    main()
