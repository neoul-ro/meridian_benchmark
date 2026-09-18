#!/usr/bin/env python3
"""기하 채점 — 매칭된 예측 검출(predicted detection)의 점군을 "그 keyframe 에 실제로 보인 GT 표면" 과 비교한다.

왜 따로 두나
  score_frontend 의 P@20cm(20cm 이내 비율)·R@20cm(커버리지)는 GT 트랙(ground-truth track) 점군(구간 전체에 걸쳐 본 표면) 기준이라,
  한 프레임에서 한쪽 면만 본 예측 검출은 completeness 가 나쁘게 나온다. 여기서는 같은 프레임의 GT 표면만 쓴다.

입력 점군 (둘 다 카메라 5m 이내 — T&T 의 crop volume 에 해당)
  GT 원점   GT 라벨 픽셀(gt_labels_2d) 을 depth 로 역투영, stride 1 (모든 라벨 픽셀). 5m 에서 픽셀 간격 1.2cm < 2cm 라
            2cm 복셀 표본이 표면에서 빠짐없이 채워진다 (예전 stride 2 는 5m 에서 2.4cm 간격이라 복셀 구멍이 생겼다)
            계산은 gt_surface.LabelSurface.frame · raw (3D 매칭의 GT 표면과 같은 코드, 수정 ⑤). pose 는 prefetch_poses 로 한 번에
            캐시 — 프레임마다 부른 값과 비트 단위로 같다 (수정 ④ 1236 keyframe · test_score_geometry 9b 가 옛 구현 사본과 다시 대조)
  매칭 관측 score_observations.csv 의 match@<τcm> (τ = 같은 폴더 score.json params.tau_m, 없으면 20cm) · ep_index · detect_credit
  예측 원점 그 관측이 발행한 점 (frontend 격자 192x256 = 2.5px 간격)

지표
  이름·정의 출처 = Occupancy Networks (Mescheder et al. CVPR 2019) im2mesh/eval.py. NICE-SLAM eval_recon.py 는 같은 식을 쓰지만
  completeness 를 'completion' 이라 부른다 (감사 D: 이름은 OccNet 을 따른다).
  accuracy     예측 → GT 최근접 거리 평균 (cm)       OccNet eval.py:110 · (같은 식) NICE-SLAM eval_recon.py:31-35
  completeness GT → 예측 최근접 거리 평균 (cm)       OccNet eval.py:99 · (같은 식, 이름 completion) NICE-SLAM eval_recon.py:38-42
  chamfer_l1   (accuracy + completeness) / 2 (cm)     OccNet eval.py:119 'chamferL1 = 0.5 * (completeness + accuracy)' — 제곱 아님
    표본 방식 차이: 두 문헌은 메시 표면에서 균일 표본(NICE-SLAM 200k 점, eval_recon.py:103,106)을 뽑는다. 여기에는 메시가
    없어서 두 원점을 2cm 복셀 중심으로 바꿔 표면 밀도를 대략 균일하게 맞춘다. 그래도 예측은 frontend 격자 간격(5m 에서 3cm)
    이라 GT(2cm)보다 성기고, 완벽한 예측도 completeness > 0 이다 → 아래 '완벽 예측 상한' 으로 그 편향을 같이 기록한다.
  P@τ · R@τ · F@τ   Tanks and Temples 공식 절차 (python_toolbox/evaluation/evaluation.py, 사본 _audit/B_2d_geometry/refs)
    :77,83   두 원점을 각각 voxel_down_sample(τ/2)  (run.py:119 voxel_size = dTau/2) — Open3D PointCloud::VoxelDownSample:
             격자 원점 = 점군 최소점 − v/2, floor, 칸 안 점 평균 (사본 _audit/fix_B_2d/refs/o3d_PointCloud.cpp:367-395)
    :173-176 precision = mean(d_pred→gt < τ), recall = mean(d_gt→pred < τ)  — strict '<' (예전 '≤ τ + 1e-9' 는 비표준)
    F = 2PR/(P+R), P+R = 0 이면 0.  τ = 5 · 10 · 20 cm. 표기는 거리 단위를 붙여 P@20cm · R@20cm · F@20cm (JSON 키는 P@20 · R@20 · F@20)
  완벽 예측 상한 (perfect_upper_bound)
    같은 관측 자리에 'GT 라벨 픽셀 자체' 를 frontend 와 같은 격자 밀도로 넣었을 때의 지표. frontend 는 depth 를 192x256 으로
    nearest 보간(sam.py pre_step: 픽셀 floor(2.5i), floor(2.5j))하므로 그 픽셀의 라벨·depth 를 정확한 픽셀 좌표로 역투영한다.
    마스크·기하 오차가 0 인 예측이 받는 점수 = 지표가 표본 밀도 때문에 낼 수 있는 최선. CSV 에는 perfect_* 열.

건너뜀 (감사 C 결함 10 — 예전에는 말없이 뺐다). 아래 순서로 처음 걸린 사유 하나로만 센다
  ignored_match         score_observations.csv 의 detect_credit = 0 — 제외 매칭(ignored match): present 아닌 GT(그 keyframe 라벨
                        5m 이내 < 1600px)에 매칭 (26/09/17 결정 ④). 채점하지 않는다. 열이 없는 옛 CSV 는 1 로 본다
                        옛 이름 outside_detect_window 는 같은 값의 deprecated 별칭으로 skipped_obs 에 남긴다 (n_skipped 에는 한 번만)
  label_png_missing     그 keyframe 라벨 PNG 없음
  gt_label_empty        매칭된 에피소드의 라벨 픽셀이 그 프레임에 없음 (3D 매칭은 됐지만 2D 라벨에서 void 로 빠진 경우 등)
  gt_no_depth_in_range  라벨 픽셀은 있으나 depth 있고 5m 이내인 픽셀이 없음
  pred_empty_in_range   예측 점이 5m 이내에 없음
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS  # noqa: E402,F401
from meridian_benchmark.uhumans2 import UH2Sequence  # noqa: E402
import gt_surface as GS  # noqa: E402

VOX = 0.02
TAUS = (0.05, 0.10, 0.20)
EPS = 1e-9          # deprecated(폐기 예정): 예전 '≤ τ + EPS' 경계용. 지금은 T&T 대로 strict '<' 라 쓰지 않는다
CROP = 40
GH, GW, CELL = 192, 256, 2.5
SKIP_REASONS = ('ignored_match', 'label_png_missing', 'gt_label_empty', 'gt_no_depth_in_range', 'pred_empty_in_range')
DEPRECATED_SKIP_ALIASES = {'outside_detect_window': 'ignored_match'}     # 옛 키 → 같은 값 (수정 ⑤, n_skipped 에 안 더함)
PERFECT_COLS = ('accuracy_cm', 'completeness_cm', 'chamfer_l1_cm', 'F@5', 'F@10', 'F@20')


def voxelize(p):
    """2cm 복셀 중심 (중복 제거) — accuracy/completeness 표본."""
    if not len(p):
        return np.zeros((0, 3))
    return (np.unique(np.floor(p / VOX).astype(np.int64), axis=0) + 0.5) * VOX


def voxel_down_sample(p, v):
    """Open3D PointCloud::VoxelDownSample 과 같은 결과: 원점 = min − v/2, floor 인덱스, 칸 안 점 평균."""
    p = np.asarray(p, np.float64)
    if not len(p):
        return np.zeros((0, 3))
    key = np.floor((p - (p.min(0) - v / 2)) / v).astype(np.int64)
    span = key.max(0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    _, inv = np.unique(flat, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv).astype(np.float64)
    return np.stack([np.bincount(inv, weights=p[:, k]) for k in range(3)], 1) / cnt[:, None]


def tnt_prf(pred_raw, gt_raw, tau):
    """Tanks and Temples evaluation.py:77,83,173-176 — (P, R, F)."""
    a = voxel_down_sample(pred_raw, tau / 2); b = voxel_down_sample(gt_raw, tau / 2)
    d1 = cKDTree(b).query(a, k=1)[0]; d2 = cKDTree(a).query(b, k=1)[0]
    P = float((d1 < tau).mean()); R = float((d2 < tau).mean())
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def geometry_metrics(pred, gt, taus=TAUS, pred_raw=None, gt_raw=None):
    """pred, gt: (N,3) 표본점(보통 2cm 복셀 중심) → accuracy/completeness/Chamfer-L1.
    pred_raw, gt_raw: F@τ 용 원점 (없으면 pred, gt 를 원점으로 쓴다). 어느 쪽이든 비면 None."""
    if not len(pred) or not len(gt):
        return None
    d_pg = cKDTree(gt).query(pred, k=1)[0]
    d_gp = cKDTree(pred).query(gt, k=1)[0]
    out = dict(accuracy_cm=float(d_pg.mean() * 100), completeness_cm=float(d_gp.mean() * 100))
    out['chamfer_l1_cm'] = (out['accuracy_cm'] + out['completeness_cm']) / 2
    pr = pred if pred_raw is None else pred_raw
    gr = gt if gt_raw is None else gt_raw
    for t in taus:
        P, R, F = tnt_prf(pr, gr, t)
        k = int(round(t * 100))
        out[f'P@{k}'] = P; out[f'R@{k}'] = R; out[f'F@{k}'] = F
    return out


def _summ(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) not in (None, '')]
        if v:
            out[k] = dict(mean=round(float(np.mean(v)), 4), median=round(float(np.median(v)), 4))
    return out


def match_column(run_dir, default_tau=0.2):
    """score_observations.csv 옆 score.json 의 params.tau_m → ('match@<τcm>', τ). 없거나 못 읽으면 기본 20cm (옛 동작)."""
    try:
        tau = float(json.loads((Path(run_dir) / 'score.json').read_text())['params']['tau_m'])
    except (OSError, ValueError, KeyError, TypeError):
        tau = default_tau
    return f'match@{int(round(tau * 100))}', tau


def score_sequence(run_dir, seq_dir, labels_dir, max_range=5.0, stride=1, out_dir=None, log=print):
    t0 = time.time()
    run_dir = Path(run_dir); labels_dir = Path(labels_dir); out_dir = Path(out_dir or run_dir)
    seq = UH2Sequence(seq_dir)
    surf = GS.LabelSurface(seq_dir, labels_dir, max_range=max_range, voxel=VOX, seq=seq)
    f = h5py.File(run_dir / 'frontend_output.h5', 'r')
    ps = f['obs/points_start'][:]; pn = f['obs/points_num'][:]; P = f['points']
    by_kf = defaultdict(list)
    n_matched = 0
    skipped = Counter({k: 0 for k in SKIP_REASONS})
    mcol, tau_m = match_column(run_dir)
    with open(run_dir / 'score_observations.csv') as fh:
        for r in csv.DictReader(fh):
            if r.get(mcol) not in (None, '', '0') and r.get('ep_index') not in (None, ''):
                n_matched += 1
                if r.get('detect_credit') not in (None, '') and int(float(r['detect_credit'])) == 0:
                    skipped['ignored_match'] += 1              # 다른 사유보다 먼저 — 겹쳐 세지 않는다
                    continue
                by_kf[(int(r['kf']), int(r['frame']))].append(r)
    surf.prefetch_poses(sorted({fr for _, fr in by_kf}))
    gi = np.floor(np.arange(GH) * CELL).astype(int); gj = np.floor(np.arange(GW) * CELL).astype(int)   # sam.py nearest
    GV, GU = np.meshgrid(gi, gj, indexing='ij')

    rows = []
    for (k, fr), obs in sorted(by_kf.items()):
        try:
            fg = surf.frame(fr, stride)
        except FileNotFoundError:
            skipped['label_png_missing'] += len(obs)
            continue
        lab, dep, K, R, t = fg.lab, fg.dep, fg.K, fg.R, fg.t
        Lg = lab[GV, GU]; Zg = dep[GV, GU]
        cam_g = GS.backproject(GU, GV, Zg, K)
        ok_g = (Zg > 0) & (np.linalg.norm(cam_g, axis=-1) <= max_range)
        present = set(int(x) for x in np.unique(lab))
        for r in obs:
            o = int(r['obs']); e = int(r['ep_index'])
            if e + 1 not in present:
                skipped['gt_label_empty'] += 1
                continue
            gt_raw = surf.raw(fg, e)
            if not len(gt_raw):
                skipped['gt_no_depth_in_range'] += 1
                continue
            p = P[ps[o]:ps[o] + pn[o]].astype(np.float64)
            p = p[np.linalg.norm(p - t, axis=1) <= max_range]
            if not len(p):
                skipped['pred_empty_in_range'] += 1
                continue
            gt = voxelize(gt_raw); pv = voxelize(p)
            m = geometry_metrics(pv, gt, pred_raw=p, gt_raw=gt_raw)
            pp = cam_g[ok_g & (Lg == e + 1)] @ R.T + t
            mp = geometry_metrics(voxelize(pp), gt, pred_raw=pp, gt_raw=gt_raw) if len(pp) else None
            rows.append(dict(kf=k, frame=fr, obs=o, tracklet_id=int(r['tracklet_id']), ep_index=e,
                             n_pred_vox=int(len(pv)), n_gt_vox=int(len(gt)),
                             **{kk: round(v, 4) for kk, v in m.items()},
                             **{f'perfect_{kk}': (round(mp[kk], 4) if mp else '') for kk in PERFECT_COLS}))
    metric_keys = [k for k in rows[0] if k not in ('kf', 'frame', 'obs', 'tracklet_id', 'ep_index', 'n_pred_vox', 'n_gt_vox')
                   and not k.startswith('perfect_')] if rows else []
    summary = _summ(rows, metric_keys)
    perfect_rows = [{kk: r[f'perfect_{kk}'] for kk in PERFECT_COLS} for r in rows if r['perfect_F@20'] != '']
    skipped_out = dict(skipped)
    skipped_out.update({old: skipped[new] for old, new in DEPRECATED_SKIP_ALIASES.items()})   # deprecated 별칭 (같은 값)
    out = dict(run=str(run_dir),
               params=dict(voxel_m=VOX, taus_m=list(TAUS), max_range_m=max_range, stride=stride,
                           gt='같은 keyframe GT 라벨 픽셀 역투영 (stride 픽셀 간격) — gt_surface.LabelSurface.frame/raw', chamfer='L1 비제곱 평균',
                           match_column=mcol, match_tau_m=tau_m,
                           skip_ignored_match='detect_credit = 0: matched GT not present at that keyframe (label px within 5m < 1600)',
                           deprecated_skip_aliases=dict(DEPRECATED_SKIP_ALIASES),
                           accuracy_completeness='2cm 복셀 중심 표본의 최근접 거리 평균 (NICE-SLAM eval_recon.py · OccNet eval.py 정의, '
                                                 '메시 균일 표본 대신 복셀 표본)',
                           fscore='Tanks and Temples evaluation.py:77,83,173-176 — 두 원점 τ/2 voxel_down_sample(Open3D 식), d < τ',
                           labels=str(labels_dir)),
               n_obs=len(rows), summary=summary, seconds=None,
               n_matched_obs=n_matched, n_skipped=int(sum(skipped[k] for k in SKIP_REASONS)), skipped_obs=skipped_out,
               perfect_upper_bound=dict(
                   definition='GT 라벨 픽셀 자체를 frontend 격자 밀도(192x256, 픽셀 floor(2.5i), floor(2.5j))로 뽑아 예측으로 넣은 값 — '
                              '표본 밀도 때문에 지표가 낼 수 있는 최선',
                   n_obs=len(perfect_rows), summary=_summ(perfect_rows, PERFECT_COLS)))
    out['seconds'] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'score_geometry.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if rows:
        with open(out_dir / 'score_geometry.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if out['n_skipped']:
        log(f'[geometry] 매칭 관측 {n_matched} 중 {out["n_skipped"]} 건너뜀 {dict((k, skipped[k]) for k in SKIP_REASONS if skipped[k])}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--seq', required=True); ap.add_argument('--labels', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    r = score_sequence(a.run, a.seq, a.labels, out_dir=a.out)
    print(json.dumps(dict(n_obs=r['n_obs'], n_skipped=r['n_skipped'], **r['summary']), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
