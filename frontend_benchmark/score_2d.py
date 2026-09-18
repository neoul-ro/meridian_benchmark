#!/usr/bin/env python3
"""segment 2D 채점 — keyframe 마다 frontend SAM 마스크를 GT 인스턴스 마스크와 이미지 위에서 비교한다.

입력  frontend_output.h5 의 obs/mask_bits (192x256 격자) · gt_labels_2d.py 가 만든 GT 라벨 PNG (640x480)
비교 격자  frontend 마스크 격자(192x256)에서 한다. GT 라벨은 격자 칸 중심 픽셀을 뽑아 맞춘다 (칸 1개 = 원본 2.5x2.5px)

판정 (keyframe 하나 안에서). 표준 = Kirillov et al. "Panoptic Segmentation", CVPR 2019 §4.1 과
공식 구현 cocodataset/panopticapi panopticapi/evaluation.py (사본 _audit/B_2d_geometry/refs/pq_compute.py, 줄 번호 동일)
  IoU        |예측 ∩ GT| / (|예측| + |GT| − |예측 ∩ GT| − |예측 ∩ void|)
             void(라벨 0) 칸은 예측에서 빼고 센다 — evaluation.py:132, 논문 §4.1 "pixels labeled void in the ground truth
             are removed from the prediction and do not affect IoU computation"
  매칭       θ 마다 따로: IoU > θ (strict, evaluation.py:134 'iou > 0.5') 인 쌍만 남긴 행렬에서 헝가리안 1:1.
             비용 = −[IoU>θ] − IoU/(2·min(N,E)) → TP 수를 먼저 최대화하고 IoU 합은 동률 깨기용
             (StarDist stardist/matching.py:177-179 과 같은 식, 사본 _audit/fix_B_2d/refs/stardist_matching.py).
             예측끼리 안 겹치면 θ ≥ 0.5 에서 IoU>0.5 쌍이 유일해서(Kirillov 정리 1) panopticapi 결과와 같다. SAM 마스크는
             서로 겹칠 수 있어 같은 GT 에 IoU>0.5 예측이 둘 생길 수 있다 — 그때도 TP 수 최대 규칙으로 1:1 로 정한다
             (panoptic 형식은 겹침을 허용하지 않아 공식 구현에 이 경우 규칙이 없다). θ = 0.25 / 0.5 / 0.75, 기본 0.5
  채점 대상(eligible) GT   면적 ≥ area_min(256칸 = 원본 1600px = frontend sam.AREA_MIN). 나머지는 제외(ignored) — 작은 GT 에 매칭된
             예측 검출(predicted detection)은 세지 않음 (제외 매칭 tp_small)
  GT 상태    값(표시 이름, terms.py GT2D):
             tp · merged = 과소분할(under-segmentation): 한 예측이 GT 2개 이상을 각각 50% 이상 덮음
             · split = 과다분할(over-segmentation): 예측 2개 이상이 각자 50% 이상 이 GT 안에 있고, 그 조각들의 합집합이 GT 의 50% 이상을 덮음
             · low_iou = Loc(localization error): 매칭 안 됨 · 최고 IoU ∈ [tide_rules.LOC_MIN_IOU, θ] = [0.1, 0.5] (양쪽 포함, 0.1 = TIDE 기본값 quantify.py:428)
               TIDE quantify.py:237 'bg_thresh <= iou <= pos_thresh' — 3D match3d 와 같은 함수 tide_rules.is_loc (수정 ⑤ E3).
               (픽셀 라벨이 배타적이라 매칭 안 된 GT 의 최고 IoU 는 θ 를 넘지 않는다 — 위쪽 경계는 3D 와 모양을 맞추려는 것)
               예전 '> min θ' (0.25 미포함)는 3D 의 [0.25, 0.5) 와 경계값에서 달랐다 (수정 ⑤ E3 · F2 에서 통일)
             · missed = 미검출(Miss): 위 어디에도 해당 안 됨. merged·split·low_iou·missed 를 합친 것이 FN 이다 (missed 만이 FN 전체가 아님)
             split 의 '합집합 50%' 조건: 조각이 GT 의 작은 모서리만 덮는 경우까지 split 로 부르지 않기 위해 넣었다.
             Caicedo et al. 2019 (unet4nuclei evaluation.py get_splits_and_merges) 는 'IoU > 0.1 인 짝이 2개 이상' 이면
             split/merge 로 세고 덮는 정도를 보지 않는다. 여기서는 그 규칙과 달리 (1) 짝 기준이 IoU 가 아니라 '예측이
             GT 안에 50% 이상' (작은 조각도 잡는다) (2) 조각 합집합 ≥ 50% 를 요구한다. 그래서 unet4nuclei 수와 1:1 로
             비교되지 않는다. 합집합이 50% 미만이면 low_iou / missed 로 떨어진다.
  예측 상태  tp · tp_small = 제외 매칭(ignored match, 작은 GT 와 매칭) · ignore = void 겹침 제외(칸의 절반 초과가 void — evaluation.py:161 '> 0.5')
             · fp = 오검출(FP)
  PQ         SQ(매칭 쌍 평균 IoU) × RQ(= TP / (TP + ½FP + ½FN)) — evaluation.py:65-67. 클래스는 하나(물체)로 본다
  사람/정적  GT 는 is_human 으로 나눈다. 예측은 매칭된 GT 의 그룹, 매칭 안 됐으면 가장 많이 겹친 GT 의 그룹
             (겹친 칸 수가 같으면 그룹별 겹친 칸 합 다수결, GT 와 전혀 안 겹치면 = 전부 void → 정적 그룹에 두지만
             ignore 라 FP 로 세지 않는다). panopticapi 가 FP 를 예측의 category_id 로 세는 것(evaluation.py:152-163)에
             해당한다 — 예측에 클래스가 없으니 겹친 GT 로 정한다.
  난이도     gt_difficulty_2d.py 의 등급 Easy·Moderate·Hard (KITTI 기준 변형, 누적)별 재현율. labels 폴더에 difficulty.npz 가 있으면 붙는다

요약 키 (score_2d.json 의 all / static / human)
  n_gt_instances n_pred_masks recall precision SQ RQ PQ gt_status pred_status
  mean_best_iou_fn      매칭 안 된 채점 대상 GT(= FN: merged·split·low_iou·missed 전부)의 최고 IoU 평균
  mean_best_iou_missed  deprecated(폐기 예정) — 이름과 달리 missed 만이 아니라 FN 전부의 평균이다.
                        mean_best_iou_fn 과 같은 값. 기존 소비자 호환용으로만 남긴다
  n_tp n_fp n_fn        PQ 계산에 들어간 수 (θ = 기본값)
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS  # noqa: E402,F401
import tide_rules as TR  # noqa: E402

GH, GW, CELL = 192, 256, 2.5
THETAS = (0.25, 0.5, 0.75)
HUMAN_COLORS = ('23d5ea',)
IGNORE_FRAC = 0.5          # void 비율이 이 값을 '넘으면' ignore (evaluation.py:161)
NEW_CSV_COLS = ('split_cover', 'assigned_ep', 'group')   # 26/09/17 표준화 때 추가한 열 — 기존 열 위치를 안 바꾸려고 맨 뒤에 쓴다
_VI = (np.arange(GH) * CELL + CELL / 2).astype(int)
_UI = (np.arange(GW) * CELL + CELL / 2).astype(int)


def gt_grid(label_png):
    """640x480 라벨 PNG → 192x256 격자 (값 = GT 트랙 ep_index+1, 0 = void)."""
    lab = cv2.imread(str(label_png), cv2.IMREAD_UNCHANGED)
    if lab is None or lab.shape != (480, 640):
        raise IOError(f'GT 라벨 읽기 실패 또는 크기 이상: {label_png}')
    return lab[_VI][:, _UI].astype(np.int32)


def match_at(iou, th):
    """θ 하나의 1:1 할당. IoU > θ 인 쌍만 후보, TP 수 최대 → IoU 합 최대 (StarDist matching.py:177-179, strict >)."""
    N, E = iou.shape
    if not (N and E):
        return []
    ok = iou > th
    if not ok.any():
        return []
    cost = -ok.astype(float) - iou / (2 * min(N, E))
    rr, cc = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rr, cc) if ok[r, c]]


def score_kf(G, M, area_min=256, theta=0.5, thetas=THETAS):
    """keyframe 하나. G (H,W) int (0=무시, k=에피소드+1) · M (N,H,W) bool 예측 마스크.
    → dict(gt=[행...], pred=[행...], matches={θ: [(pred, gt_col), ...]})
    pred 행의 assigned_ep = 매칭된 에피소드, 없으면 가장 많이 겹친 에피소드 (없으면 -1) — 사람/정적 분리에 쓴다."""
    ids, b = np.unique(G[G > 0], return_counts=True)
    E, N = len(ids), len(M)
    col = {int(g): j for j, g in enumerate(ids)}
    a = M.reshape(N, -1).sum(1) if N else np.zeros(0, int)
    inter = np.zeros((N, E))
    ign = np.zeros(N)
    for k in range(N):
        vals, cnt = np.unique(G[M[k]], return_counts=True)
        for val, c in zip(vals, cnt):
            if val == 0:
                ign[k] = c
            else:
                inter[k, col[int(val)]] = c
    # panopticapi evaluation.py:132 — union = pred + gt − inter − (pred ∩ void)
    union = a[:, None] + b[None, :] - inter - ign[:, None] if N and E else np.zeros((N, E))
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    elig = b >= area_min

    matches = {th: match_at(iou, th) for th in sorted(set(thetas) | {theta})}
    m_pred = {r: c for r, c in matches[theta]}
    m_gt = {c: r for r, c in matches[theta]}

    cover_gt = inter / np.maximum(b[None, :], 1)                 # 예측 k 가 GT j 를 덮는 비율
    inside_pred = inter / np.maximum(a[:, None], 1)              # 예측 k 가 GT j 안에 있는 비율
    merger = (cover_gt >= 0.5).sum(1) >= 2 if E else np.zeros(N, bool)   # GT 2개 이상을 각각 절반 넘게 덮은 예측

    gt_rows = []
    for j in range(E):
        split_cov = 0.0
        frags = np.flatnonzero(inside_pred[:, j] >= 0.5) if N else np.zeros(0, int)
        if len(frags) >= 2:                                     # 조각 합집합이 GT 를 덮는 비율 (픽셀 단위 — 조각끼리 겹칠 수 있다)
            split_cov = float((M[frags].any(0) & (G == ids[j])).sum() / b[j])
        if j in m_gt:
            st = 'tp'
        elif N and np.any(merger & (cover_gt[:, j] >= 0.5)):
            st = 'merged'
        elif len(frags) >= 2 and split_cov >= 0.5:
            st = 'split'
        elif N and TR.is_loc(iou[:, j].max(), theta):           # [E3·F2] [0.1, θ] 양쪽 포함 — 3D 와 같은 상수·비교
            st = 'low_iou'
        else:
            st = 'missed'
        best = int(np.argmax(iou[:, j])) if N else -1
        gt_rows.append(dict(ep_index=int(ids[j]) - 1, area_cells=int(b[j]), eligible=bool(elig[j]), status=st,
                            best_iou=round(float(iou[best, j]), 4) if N else 0.0,
                            matched_pred=m_gt.get(j, -1), best_pred=best,
                            **{f'tp@{th}': int(any(c == j for _, c in matches[th])) for th in matches},
                            split_cover=round(split_cov, 4)))
    pred_rows = []
    for k in range(N):
        if k in m_pred:
            st = 'tp' if elig[m_pred[k]] else 'tp_small'
        elif a[k] and ign[k] > IGNORE_FRAC * a[k]:
            st = 'ignore'
        else:
            st = 'fp'
        if k in m_pred:
            aj = m_pred[k]
        elif E and inter[k].max() > 0:
            aj = int(np.argmax(inter[k]))
        else:
            aj = -1
        pred_rows.append(dict(pred=k, area_cells=int(a[k]), ignore_frac=round(float(ign[k] / max(a[k], 1)), 4),
                              status=st, merger=bool(merger[k]),
                              matched_ep=int(ids[m_pred[k]]) - 1 if k in m_pred else -1,
                              iou=round(float(iou[k, m_pred[k]]), 4) if k in m_pred else
                              (round(float(iou[k].max()), 4) if E else 0.0),
                              assigned_ep=int(ids[aj]) - 1 if aj >= 0 else -1,
                              _inter={int(ids[j]) - 1: int(inter[k, j]) for j in np.flatnonzero(inter[k])} if E else {}))
    return dict(gt=gt_rows, pred=pred_rows, matches=matches)


def pred_group(row, ep_human):
    """예측 한 개의 그룹 ('human' / 'static'). 매칭 → 그 GT, 아니면 최대 겹침 GT, 겹침 칸 수 동률이면 그룹별 겹침 합 다수결."""
    inter = row.get('_inter') or {}
    if row['matched_ep'] >= 0:
        return 'human' if ep_human[row['matched_ep']] else 'static'
    if not inter:
        return 'static'                                         # GT 와 안 겹침 = 전부 void → ignore, FP 아님
    top = max(inter.values())
    tied = {bool(ep_human[e]) for e, v in inter.items() if v == top}
    if len(tied) == 1:
        return 'human' if tied.pop() else 'static'
    h = sum(v for e, v in inter.items() if ep_human[e]); s = sum(v for e, v in inter.items() if not ep_human[e])
    return 'human' if h > s else 'static'


def split_by_group(gt_rows, pred_rows, ep_human):
    """→ dict(static=(gt, pred), human=(gt, pred)). GT 는 is_human 열(없으면 ep_human), 예측은 pred_group."""
    out = {'static': ([], []), 'human': ([], [])}
    for r in gt_rows:
        h = r['is_human'] if 'is_human' in r else ep_human[r['ep_index']]
        out['human' if h else 'static'][0].append(r)
    for r in pred_rows:
        g = r.get('group') or pred_group(r, ep_human)
        out[g][1].append(r)
    return out


def summarize(gt_rows, pred_rows, theta=0.5, thetas=THETAS):
    g = [r for r in gt_rows if r['eligible']]
    if not g and not pred_rows:                               # 대상 없음 (예: 사람 0명 시퀀스) — 0% 가 아니라 없음
        return dict(n_gt_instances=0, n_pred_masks=0, recall=None, precision=None, SQ=None, RQ=None, PQ=None,
                    gt_status={}, pred_status={}, mean_best_iou_missed=None, mean_best_iou_fn=None, n_tp=0, n_fp=0, n_fn=0)
    tp = sum(r['status'] == 'tp' for r in g)
    fn = len(g) - tp
    fp = sum(r['status'] == 'fp' for r in pred_rows)
    ious = [r['iou'] for r in pred_rows if r['status'] == 'tp']
    sq = float(np.mean(ious)) if ious else 0.0
    rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-9) if (tp + fp + fn) else 0.0
    counted = [r for r in pred_rows if r['status'] in ('tp', 'fp')]
    fn_iou = round(float(np.mean([r['best_iou'] for r in g if r['status'] != 'tp'])), 4) if fn else None
    return dict(
        n_gt_instances=len(g), n_pred_masks=len(pred_rows),
        recall={str(th): round(sum(r[f'tp@{th}'] for r in g) / max(len(g), 1), 4) for th in thetas},
        precision=round(tp / max(len(counted), 1), 4),
        SQ=round(sq, 4), RQ=round(rq, 4), PQ=round(sq * rq, 4),
        gt_status=dict(Counter(r['status'] for r in g)),
        pred_status=dict(Counter(r['status'] for r in pred_rows)),
        mean_best_iou_missed=fn_iou,             # deprecated(폐기 예정): 실제로는 FN 전부의 평균 → mean_best_iou_fn
        mean_best_iou_fn=fn_iou,
        n_tp=int(tp), n_fp=int(fp), n_fn=int(fn))


LEVELS = ('easy', 'moderate', 'hard')


def by_level(gt_rows, thetas=THETAS):
    """Easy/Moderate/Hard (KITTI 기준 변형) 누적 등급별 재현율. 행에 level(0 Easy 1 Moderate 2 Hard 3 제외) 이 있어야 한다."""
    out = {}
    for k, name in enumerate(LEVELS):
        g = [r for r in gt_rows if r['eligible'] and 0 <= r.get('level', 9) <= k]
        out[name] = dict(n=len(g), recall={str(th): round(sum(r[f'tp@{th}'] for r in g) / len(g), 4) if g else None for th in thetas},
                         gt_status=dict(Counter(r['status'] for r in g)))
    return out


def score_sequence(run_dir, gt_h5, labels_dir, area_min=256, theta=0.5, out_dir=None, log=print):
    t0 = time.time()
    run_dir = Path(run_dir); out_dir = Path(out_dir or run_dir); labels_dir = Path(labels_dir)
    f = h5py.File(run_dir / 'frontend_output.h5', 'r')
    if 'mask_bits' not in f['obs']:
        raise SystemExit(f'{run_dir}: 마스크가 없는 옛 실행 결과 — run_frontend.py 로 다시 돌릴 것')
    kf_frame = f['kf/frame_idx'][:]; o0 = f['kf/obs_start'][:]; no = f['kf/n_obs'][:]
    tid = f['obs/tracklet_id'][:]; MB = f['obs/mask_bits']
    gh5 = h5py.File(gt_h5, 'r'); ix = gh5['index']
    ep_oid = ix['gt_object_id'][:]; ep_tid = ix['tracklet_id'][:]
    ep_human = np.array([str((lambda c: c.decode() if isinstance(c, bytes) else c)(
        gh5[f'tracklets/{int(t):05d}/_metadata/color_hex'][()])).lstrip('#') in HUMAN_COLORS for t in ep_tid])
    diff = {}
    dpath = labels_dir / 'difficulty.npz'
    if dpath.exists():
        z = np.load(dpath)
        diff = {(int(f_), int(e_)): (float(t_), float(o_), int(m_), int(l_)) for f_, e_, t_, o_, m_, l_ in
                zip(z['frame'], z['ep_index'], z['trunc'], z['occ'], z['min_dim'], z['level'])}
    all_gt, all_pred = [], []
    for k in range(len(kf_frame)):
        fi = int(kf_frame[k])
        G = gt_grid(labels_dir / f'{fi:06d}.png')
        M = np.unpackbits(MB[o0[k]:o0[k] + no[k]], axis=1)[:, :GH * GW].reshape(-1, GH, GW).astype(bool)
        r = score_kf(G, M, area_min=area_min, theta=theta)
        for row in r['gt']:
            e = row['ep_index']
            dv = diff.get((fi, e))
            if dv:
                row.update(trunc=round(dv[0], 3), occ=round(dv[1], 3), min_dim=dv[2], level=dv[3])
            row.update(kf=k, frame=fi, gt_object_id=int(ep_oid[e]), is_human=int(ep_human[e]),
                       matched_obs=int(o0[k] + row['matched_pred']) if row['matched_pred'] >= 0 else -1,
                       best_obs=int(o0[k] + row['best_pred']) if row['best_pred'] >= 0 else -1)
        for row in r['pred']:
            row.update(kf=k, frame=fi, obs=int(o0[k] + row['pred']), tracklet_id=int(tid[o0[k] + row['pred']]),
                       group=pred_group(row, ep_human))
            row.pop('_inter', None)
        all_gt += r['gt']; all_pred += r['pred']
    grp = split_by_group(all_gt, all_pred, ep_human)
    size_bins = [(256, 1024, '256-1023칸'), (1024, 4096, '1024-4095칸'), (4096, 10 ** 9, '4096칸+')]
    out = dict(
        run=str(run_dir), labels=str(labels_dir), params=dict(
            grid=[GH, GW], px_per_cell=CELL, area_min_cells=area_min, theta=theta, thetas=list(THETAS),
            matching='θ 마다 IoU>θ 인 쌍만 남긴 행렬에서 헝가리안 1:1 (TP 수 최대, IoU 합 동률 깨기 — StarDist matching.py:177)',
            pq='Kirillov et al. CVPR 2019 · panopticapi evaluation.py',
            iou='void(라벨 0) 칸을 예측에서 뺀다 (panopticapi evaluation.py:132)',
            ignore='void 비율 > 0.5 인 미매칭 예측 (evaluation.py:161)',
            split='예측 2개 이상이 각자 50% 이상 GT 안 + 조각 합집합이 GT 의 50% 이상',
            loc_min_iou=TR.LOC_MIN_IOU, low_iou=f'매칭 안 됨 · 최고 IoU ∈ [{TR.LOC_MIN_IOU}, θ] (TIDE quantify.py:237 bg <= iou <= pos, 3D 와 같은 tide_rules.is_loc)',
            group='예측 = 매칭 GT 그룹, 미매칭은 최대 겹침 GT 그룹'),
        all=summarize(all_gt, all_pred, theta),
        static=summarize(*grp['static'], theta),
        human=summarize(*grp['human'], theta),
        recall_by_size={lb: summarize([r for r in all_gt if lo <= r['area_cells'] < hi], [], theta)['recall']
                        for lo, hi, lb in size_bins},
        by_level=by_level(all_gt) if diff else None,
        seconds=round(time.time() - t0, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'score_2d.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for name, rows in (('score_2d_gt.csv', all_gt), ('score_2d_pred.csv', all_pred)):
        if rows:
            with open(out_dir / name, 'w', newline='') as fh:
                keys = list(dict.fromkeys(k for r in rows for k in r))
                keys = [k for k in keys if k not in NEW_CSV_COLS] + [k for k in NEW_CSV_COLS if k in keys]   # 새 열은 맨 뒤
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--gt', required=True); ap.add_argument('--labels', required=True)
    ap.add_argument('--area-min', type=int, default=256); ap.add_argument('--theta', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    s = score_sequence(a.run, a.gt, a.labels, a.area_min, a.theta, out_dir=a.out)
    print(json.dumps({k: s[k] for k in ('all', 'static', 'human', 'recall_by_size')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
