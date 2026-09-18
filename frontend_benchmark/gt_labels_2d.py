#!/usr/bin/env python3
"""keyframe 마다 GT 인스턴스 라벨 이미지(640x480)를 만든다 — 2D 분할 채점과 뷰어가 쓴다.

라벨 값 (uint16 PNG): 0 = void(무시 라벨: GT 트랙이 아닌 표면, 검증 안 되는 픽셀)
                     k = GT 트랙(ground-truth track) 번호(ep_index) + 1  (GT 트랙 = 코드·CSV 의 에피소드)

방법
  seg 색은 prefab(재질) 타입이지 인스턴스가 아니다. 그래서
  1. 그 프레임에 활성인 GT 트랙(first ≤ f ≤ last)의 색만 본다
  2. 같은 색 영역을 8-연결 성분으로 나눈다
  3. 성분 안 픽셀을 stride 2 로 역투영(depth 있고 카메라 5m 이내 = 표본) → 같은 색 활성 에피소드 점군까지 최근접 거리
     tol(6cm) 이내 표본(hit)이 어느 에피소드에 붙는지 투표
  4. 지지율 = hit / 표본 ≥ support_min(0.5) 일 때만 성분 단위로 채운다
       한 에피소드가 hit 의 90% 이상 → 성분 전체를 그 에피소드로 (임시 라벨)
       여럿이 나눠 가지면 → 성분 안 각 픽셀을 이미지상 가장 가까운 hit 표본의 에피소드로 (임시 라벨)
     지지율 < 0.5 → 성분을 채우지 않는다. 5m 이내 depth 픽셀은 픽셀마다 같은 색 활성 에피소드 6cm 검사로만 라벨,
       5m 밖 픽셀은 가장 가까운 hit 표본의 에피소드를 임시 라벨로 두고 5 에서 검증
  5. 픽셀 단위 검증 (임시 라벨 → 확정)
       depth 없음(0)                → void (검증할 수 없다)
       카메라 5m 밖                  → 임시 라벨 에피소드와 같은 gt_object_id 의 '모든 에피소드' 점군 합집합까지 6cm 이내면 유지,
                                      아니면 void. GT 점군은 5m 안에서 본 표면만 있으므로, 이 프레임에서 5m 밖이어도 다른 때 5m
                                      안에서 본 적이 있는 표면이면 확인된다. 같은 prefab 의 다른 물체가 한 성분으로 이어져
                                      합쳐지던 문제(감사 B 결함 3 · 감사 C 결함 2)를 막는다
       5m 이내 (지지율 ≥ 0.5 성분)  → 성분 투표 라벨 유지 (표본 검증은 3 에서 stride 2 로 했다)
  6. hit 표본이 하나도 없는 성분 → 무시
  tol·stride 는 gt_visibility.py 와 같다 (그쪽 검증: 5m 제한 없이 GT 관측 프레임 수와 중앙값 1.00 일치).
  5m 밖 검증은 감사 far_pixels.py 의 'same_obj' 판정과 같은 기준(같은 물체 합집합 6cm)이고, 그쪽의 '다른 물체가 더 가까우면
  제외' 조건은 넣지 않았다 (맞닿은 같은 prefab 물체 경계에서만 차이, 남은 위험으로 기록).

출력: <out>/<frame>.png  (640x480, 가운데 crop 기준) · <out>/meta.json
      meta.json 의 version 이 LABEL_VERSION 과 다르거나 없으면 build 가 기존 PNG 를 건너뛰지 않고 전부 다시 만든다
      (옛 알고리즘 라벨이 캐시로 조용히 재사용되는 것을 막는다)
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
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS  # noqa: E402,F401
from meridian_benchmark.uhumans2 import UH2Sequence  # noqa: E402

CROP = 40
OBJ_TREE_CACHE = 64
LABEL_VERSION = '2026-09-17 표준화 (감사 B 결함 3)'   # 라벨 알고리즘이 바뀌면 올린다 → 옛 PNG 를 재사용하지 않음
G = {}


def labels_outdated(out):
    """out 폴더의 PNG 가 다른 알고리즘 버전으로 만들어졌나 (meta.json 의 version 이 없거나 다름). PNG 가 없으면 False."""
    out = Path(out)
    if not any(out.glob('*.png')):
        return False
    try:
        return json.loads((out / 'meta.json').read_text()).get('version') != LABEL_VERSION
    except (OSError, ValueError):
        return True


def _init(seq_dir, gt_path, stride, max_range, tol, dominant, support_min=0.5):
    G['seq'] = UH2Sequence(seq_dir)
    f = h5py.File(gt_path, 'r'); ix = f['index']
    G['h5'] = f
    G['tid'] = ix['tracklet_id'][:]; G['first'] = ix['first_frame'][:]; G['last'] = ix['last_frame'][:]
    G['oid'] = ix['gt_object_id'][:]
    G['color'] = np.array([int((lambda c: c.decode() if isinstance(c, bytes) else c)(
        f[f'tracklets/{int(t):05d}/_metadata/color_hex'][()]).lstrip('#'), 16) for t in G['tid']], np.int64)
    G['K'] = G['seq'].camera_info()['K']
    G['trees'] = {}; G['obj_trees'] = {}
    G.update(stride=stride, max_range=max_range, tol=tol, dominant=dominant, support_min=support_min)


def _tree(i):
    if i not in G['trees']:
        G['trees'][i] = cKDTree(G['h5'][f'tracklets/{int(G["tid"][i]):05d}/tracklet_geometry/points'][:])
    return G['trees'][i]


def _obj_tree(oid):
    """같은 gt_object_id 의 모든 에피소드 점군 합집합."""
    oid = int(oid)
    if oid not in G['obj_trees']:
        eps = np.flatnonzero(G['oid'] == oid)
        pts = np.concatenate([G['h5'][f'tracklets/{int(G["tid"][e]):05d}/tracklet_geometry/points'][:] for e in eps])
        if 'h5' in G and len(G['obj_trees']) >= OBJ_TREE_CACHE:
            G['obj_trees'].pop(next(iter(G['obj_trees'])))
        G['obj_trees'][oid] = cKDTree(pts)
    return G['obj_trees'][oid]


def label_frame(fi):
    """→ (640x480 uint16 라벨, {ep_index: 픽셀 수})"""
    seq, K, s, tol = G['seq'], G['K'], G['stride'], G['tol']
    support_min = G.get('support_min', 0.5)
    lab = np.zeros((480, 720), np.uint16)
    act = np.flatnonzero((G['first'] <= fi) & (G['last'] >= fi))
    if not len(act):
        return lab[:, CROP:CROP + 640], {}
    seg = seq.seg_packed(fi); dep = seq.depth(fi)
    t, q = seq.camera_poses(seq.stamps_ns[[fi]], cam='left_cam'); R = Rotation.from_quat(q[0]).as_matrix()

    def backproject(uu, vv, z):
        return np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1)

    for c in np.unique(G['color'][act]):
        m = seg == c
        if not m.any():
            continue
        eps = act[G['color'][act] == c]
        comp, ncomp = ndimage.label(m, structure=np.ones((3, 3), bool))
        vv, uu = np.nonzero(m[::s, ::s]); vv = vv * s; uu = uu * s
        z = dep[vv, uu]
        ok = z > 0
        vv, uu, z = vv[ok], uu[ok], z[ok]
        cam = backproject(uu, vv, z)
        near = np.linalg.norm(cam, axis=1) <= G['max_range']
        vv, uu, cam = vv[near], uu[near], cam[near]
        if not len(cam):
            continue
        W = cam @ R.T + t[0]
        D = np.stack([_tree(e).query(W, k=1, distance_upper_bound=tol)[0] for e in eps])   # (E, S)
        best = np.argmin(D, 0); hit = np.isfinite(D.min(0))
        sc = comp[vv, uu]
        n_samp = np.bincount(sc, minlength=ncomp + 1)
        for cid in np.unique(sc[hit]):
            sel = hit & (sc == cid)
            support = sel.sum() / n_samp[cid]
            votes = np.bincount(best[sel], minlength=len(eps))
            pv, pu = np.nonzero(comp == cid)
            zp = dep[pv, pu]
            campx = backproject(pu, pv, zp)
            rng = np.linalg.norm(campx, axis=1)
            has_d = zp > 0
            far = has_d & (rng > G['max_range'])
            near_px = has_d & ~far
            prov = np.zeros(len(pv), np.int64)                  # 임시 라벨 = 에피소드 + 1 (0 = void)
            nearest_hit_ep = None
            if support < support_min or votes.max() < G['dominant'] * votes.sum():
                _, j = cKDTree(np.stack([vv[sel], uu[sel]], 1)).query(np.stack([pv, pu], 1), k=1)
                nearest_hit_ep = eps[best[sel][j]] + 1
            if support >= support_min:
                prov[:] = eps[int(votes.argmax())] + 1 if nearest_hit_ep is None else nearest_hit_ep
            else:                                               # 지지 부족 → 성분 채우기 없음, 픽셀 단위 검증만
                if near_px.any():
                    Wn = campx[near_px] @ R.T + t[0]
                    Dn = np.stack([_tree(e).query(Wn, k=1, distance_upper_bound=tol)[0] for e in eps])
                    prov[near_px] = np.where(np.isfinite(Dn.min(0)), eps[np.argmin(Dn, 0)] + 1, 0)
                prov[far] = nearest_hit_ep[far]
            prov[~has_d] = 0                                    # depth 없음 → 검증 불가 → void
            fidx = np.flatnonzero(far & (prov > 0))
            if len(fidx):                                       # 5m 밖 → 같은 물체 전 에피소드 합집합 6cm
                Wf = campx[fidx] @ R.T + t[0]
                keep = np.zeros(len(fidx), bool)
                pe = prov[fidx] - 1
                for oid in np.unique(G['oid'][pe]):
                    s_ = G['oid'][pe] == oid
                    keep[s_] = np.isfinite(_obj_tree(oid).query(Wf[s_], k=1, distance_upper_bound=tol)[0])
                prov[fidx[~keep]] = 0
            lab[pv, pu] = prov
    crop = lab[:, CROP:CROP + 640]
    ids, n = np.unique(crop[crop > 0], return_counts=True)
    return crop, {int(i) - 1: int(k) for i, k in zip(ids, n)}


def _job(args):
    fi, out = args
    p = Path(out) / f'{fi:06d}.png'
    crop, counts = label_frame(fi)
    cv2.imwrite(str(p), crop)
    return fi, counts


def build(seq_dir, gt_path, frames, out, workers=8, stride=2, max_range=5.0, tol=0.06, dominant=0.9, support_min=0.5, log=print):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frames = sorted(set(int(x) for x in frames))
    stale = labels_outdated(out)
    if stale:
        log(f'[gt2d] {out}: 라벨 알고리즘 버전이 달라 ({LABEL_VERSION} 아님) 전부 다시 만든다')
    todo = [(fi, str(out)) for fi in frames if stale or not (out / f'{fi:06d}.png').exists()]
    t0 = time.time()
    if todo:
        args = (str(seq_dir), str(gt_path), stride, max_range, tol, dominant, support_min)
        try:                       # 부모에서 먼저 한 번 연다 — 자식(Pool initializer)에서 실패하면
            _init(*args)           # Pool 이 워커를 끝없이 다시 띄워 멈추고 로그가 폭증한다 (사용성 리뷰 C4 · e17)
        except Exception as e:     # noqa: BLE001
            msg = str(e).splitlines()[0] if str(e).strip() else type(e).__name__
            raise SystemExit(f'[gt2d] 입력을 읽을 수 없습니다 (데이터셋 {seq_dir} · GT {gt_path}) — {msg}. '
                             f'FB_DATA · FB_GT 를 확인해 주세요.')
        finally:
            if 'h5' in G:
                try:
                    G['h5'].close()
                except Exception:  # noqa: BLE001
                    pass
            G.clear()
        with Pool(workers, initializer=_init, initargs=args) as pool:
            for k, _ in enumerate(pool.imap_unordered(_job, todo, chunksize=8)):
                if (k + 1) % 500 == 0:
                    log(f'[gt2d] {k + 1}/{len(todo)} · {time.time() - t0:.0f}s')
    (out / 'meta.json').write_text(json.dumps(dict(
        seq=str(seq_dir), gt=str(gt_path), stride=stride, max_range=max_range, tol=tol, dominant=dominant,
        crop=[CROP, CROP + 640], value='ep_index+1 (0 = ignore)', n_frames=len(frames),
        support_min=support_min,
        far_check='5m 밖 픽셀: 같은 gt_object_id 전 에피소드 점군 합집합 tol 이내만 유지',
        depth0='void', version=LABEL_VERSION), indent=2, ensure_ascii=False))
    log(f'[gt2d] {len(frames)} 프레임 (새로 만든 것 {len(todo)}) · {time.time() - t0:.0f}s')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='frontend_output.h5 가 있는 폴더 — keyframe 프레임만 만든다')
    ap.add_argument('--seq', required=True); ap.add_argument('--gt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    frames = h5py.File(Path(a.run) / 'frontend_output.h5', 'r')['kf/frame_idx'][:]
    build(a.seq, a.gt, frames, a.out, workers=a.workers)


if __name__ == '__main__':
    main()
