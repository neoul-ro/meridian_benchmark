#!/usr/bin/env python3
"""추적(tracker) 채점 — keyframe 을 timestep 으로 보고 MOT 표준 지표(CLEAR · Identity · HOTA)를 계산한다.
지표 알고리즘은 TrackEval (github.com/JonathonLuiten/TrackEval, commit 12c8791) 의 eval_sequence 와 같은 절차다.
test_trackeval_parity.py 가 합성·실제 데이터에서 TrackEval 과 1e-6 이내로 같은지 확인한다 (MT 경계만 예외, 아래).

입력  score_frontend.py 결과 — score_observations.csv 의 관측별 IoU_τ(gt_iou@<τ>) · 가장 가까운 표면 복셀 수(gt_vox@<τ>) · n_vox,
      score_kf_gt.csv 의 (keyframe, GT) 라벨 px, score.json 의 params(tau_m · iou_threshold · iou_column · vox_column · kf_gt_file)
      GT tracklet h5(에피소드 → 물체 번호 · 색). gt_visibility npz 는 인자로만 남았다(present 는 라벨 기준, 26/09/17 결정 ④).

GT 트랙   기본 = 정적 GT 물체의 **연속 가시 구간(에피소드)**. keyframe k 에서 라벨 픽셀(5m 이내·depth 유효) ≥ 1600px 이면
          "있음"(present) — score_frontend 의 present 와 같은 정의(score_kf_gt.csv px_label_5m).
          위키 Evaluation: "보낸 거 또 보냄 → 괜찮다, DA 가 해결할 문제" — 시야를 벗어났다 돌아왔을 때 새 id 를 주는 건
          frontend 의 잘못이 아니다. 그래서 id 유지는 보이는 동안(에피소드 안)만 따진다.
          unit='object' 로 주면 시퀀스 전체의 물체 단위(재방문까지 같은 id 요구, DA 까지 포함한 기준) — 참고용.
          사람은 뺀다 — 움직이는 사람이 GT 에서 여러 물체 번호로 쪼개져 있어 GT 정체성부터 믿을 수 없다.
무시 GT   그 keyframe 에 라벨 표면은 있지만 present 가 아니거나 사람인 GT (TrackEval MOT17 의 distractor 클래스에 해당).
예측 검출 keyframe 마다 score_frontend 와 같은 전처리 매칭(match3d.match: 표면 있는 GT 전부 × 관측, IoU_τ ≥ 0.5 헝가리안)을 하고
          · 무시 GT 에 매칭된 관측 → 뺀다 (TrackEval mot_challenge_2d_box.py:370-387 'remove tracker dets which match with gt dets
            which are labeled as belonging to a distractor class')
          · 매칭 안 된 관측 중 (void + 가장 가까운 표면이 무시 GT 인 복셀) / |P| > 0.5 → 뺀다 (kitti_mots.py:336-344 ignore region,
            panopticapi evaluation.py:161, ScanNet evaluate_semantic_instance.py:152-163)
          · 복셀 0 (5m 밖) → 뺀다.  나머지 = 예측 검출 → 매칭 안 된 예측 검출은 전부 FP (TrackEval clear.py:113). 중복(dup_of)도 FP.
          뺀 이유는 n_obs_excluded = {no_voxels_within_range, matched_non_present_gt, matched_human_gt, mostly_ignore_region}.
유사도    similarity[g, 관측] = IoU_τ (score_frontend 가 17 유효숫자로 쓴 gt_iou 값 그대로, 0~1). CLEAR·Identity·HOTA 모두 이 행렬 하나를 쓴다.
          α(임계) = score.json 의 iou_threshold(= min_frac) 0.5 하나 (단일 α — HOTA 의 α 적분은 하지 않는다 → HOTA_α).
          예전(감사 A 수정 ①)엔 share = gt_counts[oid] / n_vox 였다.

지표 (정의 · 출처)
  CLEAR 매칭  keyframe 마다 score = 1000·[예측 id == 그 GT 의 **바로 이전 timestep** 매칭 id] + similarity,
              similarity < α 인 칸은 0, 헝가리안 → score > 0 인 쌍만 매칭.  TrackEval clear.py:78-91,
              연속성 기억 = clear.py:62-65·80-82·104-105 (매 timestep 초기화, 없는 GT 는 기억이 지워짐).
              MOT16 §4.1.1 "matched at t−1 ... carried over to frame t".
              · 공식 구현 동작 그대로: 그 timestep 에 있는 GT 가 0 개이거나 예측 검출이 0 개면(clear.py:70-76) 매칭을
                건너뛰고 기억도 지우지 않는다. 그래서 검출이 하나도 없는 keyframe 너머로는 직전 매칭이 이어진다.
              (예전엔 score_frontend 가 이 우선권을 영구 기억으로 줬고 don't care keyframe 의 매칭으로도 갱신됐다 — 감사 A-4)
  MOTA        1 − (FN + FP + IDSW) / GT.  Bernardin & Stiefelhagen 2008 식 (TrackEval clear.py:167 과 GT>0 에서 같음)
  IDSW        매칭될 때 그 GT 의 마지막 매칭 id(몇 timestep 전이든)와 다르면 1.  MOT16 §4.1.1, clear.py:93-97
  Frag        GT 마다 "직전 timestep 에 추적 안 됨 → 지금 추적됨" 횟수 − 1 의 합.  TrackEval clear.py:102-107·122.
              GT 가 없음(<1600px)인 keyframe 도 추적 안 된 것으로 치므로, 없음 구간을 사이에 두고 다시 추적되면 1 (감사 A-5).
              py-motmetrics(num_fragmentations) 와 다른 점: py-motmetrics 는 GT 가 있는 프레임 행만 보므로 없음 구간은
              끊김으로 치지 않는다. 위 공식 구현 동작 때문에 검출 0개 keyframe 도 끊김으로 치지 않는다.
  MT/PT/ML    GT 가 있는 keyframe 중 매칭 비율 ≥ 80% / 20% 이상 80% 미만 / < 20%.  MOT16 §4.1.6 "at least 80%".
              TrackEval clear.py:119 는 MT = 비율 > 0.8 (정확히 0.8 이면 PT) — 이 경계 하나만 다르다.
              정확히 80% 인 트랙 수를 MT_ratio_eq_80 으로 같이 낸다 (TrackEval MT = MT − MT_ratio_eq_80).
  IDF1        2·IDTP / (2·IDTP + IDFP + IDFN).  Ristani et al. 2016 식 3~10: (GT 트랙, 예측 id) 쌍의 공동 출현 수 =
              similarity ≥ α 인 모든 timestep 수 (1:1 CLEAR 매칭 여부와 무관) → 트랙 단위 헝가리안.
              TrackEval identity.py:53-57(potential_matches_count)·63-85. 예전엔 CLEAR 1:1 매칭만 셌다 (감사 A-1).
              IDTP = max Σ potential (직사각 헝가리안). identity.py 의 (G+T)² 비용 행렬 최적해와 값이 같다:
              그 비용 = Σ|g| + Σ|t| − 2·Σ_matched potential 이라 비용 최소 ⇔ Σ potential 최대.
  HOTA        √(DetA · AssA).  Luiten et al. IJCV 2021 식 13~19. TrackEval hota.py:47-116 절차:
              ① 모든 timestep 의 similarity 로 정규화 공동 출현(sim_iou)을 누적 → global alignment score (hota.py:52-68)
              ② timestep 마다 global_alignment × similarity 로 헝가리안을 **다시** 하고 similarity ≥ α 인 쌍만 TP (hota.py:84-101)
              ③ AssA = Σ_TP A(c) / |TP|, A = TPA/(TPA+FNA+FPA) (hota.py:105-108), DetA = TP/(TP+FN+FP).
              HOTA 의 TP/FN/FP 는 CLEAR 의 것과 다를 수 있어 HOTA_TP·HOTA_FN·HOTA_FP 로 따로 낸다.
              예전엔 CLEAR 매칭을 그대로 AssA 에 썼다 (감사 A-2).
  분모 0      MOTA(GT 0) · IDF1 · DetA 가 0/0 이면 null. TP 가 0 이면 AssA 는 null(평균할 TP 없음), HOTA 는 식 13 대로 0.
              (TrackEval 은 이때 AssA 에 0 을 쓴다.)
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS  # noqa: E402,F401
import match3d as M  # noqa: E402

HUMAN_COLORS = ('23d5ea',)
FEPS = np.finfo('float').eps                          # TrackEval 과 같은 비교 여유 (clear.py:82·86, hota.py:59·92)


def parse_counts(s):
    """score_frontend.format_counts 의 역: 'key:count;key:count' → {key: count}."""
    return {int(a): int(b) for a, b in (x.split(':') for x in s.split(';') if x)} if s else {}


def data_from_timesteps(steps):
    """[(GT 키 목록, 예측 키 목록, similarity [len(GT) × len(예측)])] → TrackEval eval_sequence 입력 dict.
    키는 hashable 아무거나. 번호는 처음 나온 순서로 0.. (TrackEval 데이터셋의 연속 번호 relabel 과 같은 형태).
    같은 timestep 안에서 키가 겹치면 TrackEval _check_unique_ids 처럼 거부한다."""
    gmap, tmap = {}, {}
    data = dict(gt_ids=[], tracker_ids=[], similarity_scores=[])
    for t, (gk, tk, sim) in enumerate(steps):
        if len(set(gk)) != len(gk) or len(set(tk)) != len(tk):
            raise ValueError(f'timestep {t}: 같은 id 가 두 번 나옴 (GT {list(gk)} / 예측 {list(tk)})')
        data['gt_ids'].append(np.array([gmap.setdefault(g, len(gmap)) for g in gk], int))
        data['tracker_ids'].append(np.array([tmap.setdefault(p, len(tmap)) for p in tk], int))
        data['similarity_scores'].append(np.asarray(sim, float).reshape(len(gk), len(tk)))
    data.update(num_timesteps=len(steps), num_gt_ids=len(gmap), num_tracker_ids=len(tmap),
                num_gt_dets=int(sum(len(x) for x in data['gt_ids'])),
                num_tracker_dets=int(sum(len(x) for x in data['tracker_ids'])),
                gt_keys=list(gmap), tracker_keys=list(tmap))
    return data


def clear_metrics(data, threshold=0.5):
    """CLEAR MOT — TrackEval clear.py:37-129 와 같은 절차. 반환 (지표 dict, 타임라인 {GT 번호: [(t, 예측 번호|None)]})."""
    G = data['num_gt_ids']
    tp = fn = fp = idsw = 0
    gt_id_count = np.zeros(G); gt_matched_count = np.zeros(G); gt_frag_count = np.zeros(G)
    prev_tracker_id = np.full(G, np.nan)                  # IDSW 채점용: 마지막 매칭 id (clear.py:64)
    prev_timestep_tracker_id = np.full(G, np.nan)         # 매칭용 연속성: 바로 이전 timestep 의 매칭 id (clear.py:65)
    timeline = defaultdict(list)
    for t, (g, tr) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        if len(g) == 0:                                   # clear.py:70-72 — 기억은 그대로
            fp += len(tr)
            continue
        if len(tr) == 0:                                  # clear.py:73-76 — 기억은 그대로
            fn += len(g)
            gt_id_count[g] += 1
            for gi in g:
                timeline[int(gi)].append((t, None))
            continue
        sim = data['similarity_scores'][t]
        score_mat = (tr[np.newaxis, :] == prev_timestep_tracker_id[g[:, np.newaxis]])
        score_mat = 1000 * score_mat + sim
        score_mat[sim < threshold - FEPS] = 0
        rows, cols = linear_sum_assignment(-score_mat)
        ok = score_mat[rows, cols] > 0 + FEPS
        rows, cols = rows[ok], cols[ok]
        mg, mt = g[rows], tr[cols]
        prev = prev_tracker_id[mg]
        idsw += int(np.sum(np.logical_not(np.isnan(prev)) & np.not_equal(mt, prev)))
        gt_id_count[g] += 1
        gt_matched_count[mg] += 1
        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[mg] = mt
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[mg] = mt
        currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
        gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)
        tp += len(mg); fn += len(g) - len(mg); fp += len(tr) - len(mg)
        m = dict(zip(mg.tolist(), mt.tolist()))
        for gi in g:
            timeline[int(gi)].append((t, m.get(int(gi))))
    has = gt_id_count > 0
    mc, gc = gt_matched_count[has], gt_id_count[has]      # 정수값 float — 5·m ≥ 4·c 는 비율 ≥ 0.8 과 정확히 같다
    n_mt = int(np.sum(5 * mc >= 4 * gc))                  # MOT16 §4.1.6 "at least 80%" (TrackEval clear.py:119 는 >)
    n_pt = int(np.sum(5 * mc >= gc)) - n_mt               # 20% 이상 (clear.py:120 과 같음)
    gt_n = int(data['num_gt_dets'])
    return dict(
        TP=tp, FN=fn, FP=fp, IDSW=idsw,
        Frag=int(np.sum(gt_frag_count[gt_frag_count > 0] - 1)),
        MT=n_mt, PT=n_pt, ML=G - n_mt - n_pt, MT_ratio_eq_80=int(np.sum(5 * mc == 4 * gc)),
        MOTA=(1 - (fn + fp + idsw) / gt_n) if gt_n else None,
    ), timeline


def identity_metrics(data, threshold=0.5):
    """ID 지표 — Ristani et al. 2016 식 3~10, TrackEval identity.py:31-89."""
    gt_n, pr_n = int(data['num_gt_dets']), int(data['num_tracker_dets'])
    idtp = 0
    if gt_n and pr_n:
        pot = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        for g, tr, sim in zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores']):
            mg, mt = np.nonzero(np.greater_equal(sim, threshold))       # identity.py:55-57 (공동 출현 = 전부)
            pot[g[mg], tr[mt]] += 1
        rows, cols = linear_sum_assignment(-pot)
        idtp = int(round(pot[rows, cols].sum()))
    idfn, idfp = gt_n - idtp, pr_n - idtp
    den = 2 * idtp + idfp + idfn
    return dict(IDTP=idtp, IDFP=idfp, IDFN=idfn, IDF1=(2 * idtp / den) if den else None,
                IDR=(idtp / (idtp + idfn)) if (idtp + idfn) else None, IDP=(idtp / (idtp + idfp)) if (idtp + idfp) else None)


def hota_metrics(data, alpha=0.5):
    """HOTA 단일 α — Luiten et al. 2021 식 13~19, TrackEval hota.py:24-116 (array_labels = [α])."""
    gt_n, pr_n = int(data['num_gt_dets']), int(data['num_tracker_dets'])
    tp = fn = fp = 0
    ass_a = ass_re = ass_pr = None
    if not gt_n or not pr_n:                               # hota.py:36-45
        fn, fp = gt_n, pr_n
    else:
        G, Tn = data['num_gt_ids'], data['num_tracker_ids']
        potential = np.zeros((G, Tn)); gt_id_count = np.zeros((G, 1)); tracker_id_count = np.zeros((1, Tn))
        steps = list(zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores']))
        for g, tr, sim in steps:                           # hota.py:53-65
            denom = sim.sum(0)[np.newaxis, :] + sim.sum(1)[:, np.newaxis] - sim
            sim_iou = np.zeros_like(sim)
            mask = denom > 0 + FEPS
            sim_iou[mask] = sim[mask] / denom[mask]
            potential[g[:, np.newaxis], tr[np.newaxis, :]] += sim_iou
            gt_id_count[g] += 1
            tracker_id_count[0, tr] += 1
        global_alignment = potential / (gt_id_count + tracker_id_count - potential)   # hota.py:68
        matches = np.zeros_like(potential)
        for g, tr, sim in steps:                           # hota.py:72-101
            if len(g) == 0:
                fp += len(tr)
                continue
            if len(tr) == 0:
                fn += len(g)
                continue
            score_mat = global_alignment[g[:, np.newaxis], tr[np.newaxis, :]] * sim
            rows, cols = linear_sum_assignment(-score_mat)
            ok = sim[rows, cols] >= alpha - FEPS
            rows, cols = rows[ok], cols[ok]
            tp += len(rows); fn += len(g) - len(rows); fp += len(tr) - len(rows)
            matches[g[rows], tr[cols]] += 1
        if tp:                                             # hota.py:105-112
            ass_a = float(np.sum(matches * (matches / np.maximum(1, gt_id_count + tracker_id_count - matches))) / tp)
            ass_re = float(np.sum(matches * (matches / np.maximum(1, gt_id_count))) / tp)
            ass_pr = float(np.sum(matches * (matches / np.maximum(1, tracker_id_count))) / tp)
    den = tp + fn + fp
    det_a = tp / den if den else None
    hota = (float(np.sqrt(det_a * ass_a)) if ass_a is not None else 0.0) if den else None
    return dict(DetA=det_a, AssA=ass_a, HOTA=hota, HOTA_TP=tp, HOTA_FN=fn, HOTA_FP=fp,
                DetRe=tp / (tp + fn) if (tp + fn) else None, DetPr=tp / (tp + fp) if (tp + fp) else None,
                AssRe=ass_re, AssPr=ass_pr)


def mot_metrics(data, threshold=0.5):
    """data (data_from_timesteps / build_data) → (지표 dict — 반올림 없음, 타임라인 {GT 키: [(timestep, 예측 키|None)]})."""
    clear, tl = clear_metrics(data, threshold)
    ident = identity_metrics(data, threshold)
    hota = hota_metrics(data, threshold)
    gk, tk = data['gt_keys'], data['tracker_keys']
    timeline = {gk[g]: [(t, None if p is None else tk[p]) for t, p in seq] for g, seq in tl.items()}
    ids_per = [len({p for _, p in seq if p is not None}) for seq in timeline.values() if any(p is not None for _, p in seq)]
    m = dict(
        n_gt_tracks=int(data['num_gt_ids']), n_pred_ids=int(data['num_tracker_ids']), gt_detections=int(data['num_gt_dets']),
        TP=clear['TP'], FN=clear['FN'], FP=clear['FP'], IDSW=clear['IDSW'], Frag=clear['Frag'],
        MT=clear['MT'], PT=clear['PT'], ML=clear['ML'], MOTA=clear['MOTA'],
        IDTP=ident['IDTP'], IDFP=ident['IDFP'], IDFN=ident['IDFN'], IDF1=ident['IDF1'],
        DetA=hota['DetA'], AssA=hota['AssA'], HOTA=hota['HOTA'],
        mean_ids_per_track=float(np.mean(ids_per)) if ids_per else None,
        # --- 26/09/17 추가 ---
        pred_detections=int(data['num_tracker_dets']), MT_ratio_eq_80=clear['MT_ratio_eq_80'],
        IDR=ident['IDR'], IDP=ident['IDP'],
        HOTA_TP=hota['HOTA_TP'], HOTA_FN=hota['HOTA_FN'], HOTA_FP=hota['HOTA_FP'],
        DetRe=hota['DetRe'], DetPr=hota['DetPr'], AssRe=hota['AssRe'], AssPr=hota['AssPr'])
    return m, timeline


def build_data(rows, kf_gt, n_kf, ep_oid, human_obj, area_min_px, unit, iou_col, vox_col, threshold=M.IOU_THRESHOLD):
    """score_observations.csv 행 · score_kf_gt.csv 행 → TrackEval 입력 dict (모듈 docstring 의 GT 트랙 · 무시 GT · 예측 검출 · 유사도).
    반환 dict 에 excluded = {이유: 관측 수} · removed = [(keyframe, tracklet_id, 이유)] 를 붙인다."""
    gts_at = defaultdict(list)                              # keyframe → [(ep, 라벨 px)] (표면 있는 GT 전부)
    for r in kf_gt:
        gts_at[int(r['kf'])].append((int(r['ep_index']), int(r['px_label_5m'])))
    obs_at = defaultdict(list)                              # keyframe → [(tid, n_vox, {ep: IoU}, {ep: 가장 가까운 복셀 수})] (CSV 순서)
    for r in rows:
        obs_at[int(r['kf'])].append((int(r['tracklet_id']), int(r['n_vox']), M.parse_values(r[iou_col]), parse_counts(r[vox_col])))
    excluded = defaultdict(int)
    removed = []
    steps = []
    for k in range(n_kf):
        cand = sorted(gts_at[k])                            # score_frontend 와 같은 GT 순서 (에피소드 번호 오름차순)
        eps = [e for e, _ in cand]
        human = [int(ep_oid[e]) in human_obj for e in eps]
        valid = [px >= area_min_px and not h for (_, px), h in zip(cand, human)]
        obs = obs_at[k]
        iou = np.array([[o[2].get(e, 0.0) for o in obs] for e in eps], np.float64).reshape(len(eps), len(obs))
        rr, cc = M.match(iou, threshold)                     # 전처리 매칭 = score_frontend 의 주 τ 매칭 (같은 함수·같은 행렬)
        matched = {int(c): int(r) for r, c in zip(rr, cc)}
        tk, cols = [], []
        for i, (tid, nv, ious, vox) in enumerate(obs):
            why = None
            if nv <= 0:
                why = 'no_voxels_within_range'
            elif i in matched:
                j = matched[i]
                if not valid[j]:
                    why = 'matched_human_gt' if human[j] else 'matched_non_present_gt'
            else:
                near = np.array([vox.get(e, 0) for e in eps], np.int64)
                void = nv - int(sum(vox.values()))
                status, _ = M.unmatched_class(nv, void, near, ~np.array(valid, bool))
                if status == 'excluded':
                    why = 'mostly_ignore_region'
            if why:
                excluded[why] += 1; removed.append((k, tid, why))
                continue
            tk.append(tid); cols.append(ious)
        gsel = [j for j in range(len(eps)) if valid[j]]
        gk = [eps[j] if unit == 'episode' else int(ep_oid[eps[j]]) for j in gsel]
        sim = np.array([[c.get(eps[j], 0.0) for c in cols] for j in gsel], np.float64).reshape(len(gsel), len(cols))
        steps.append((gk, tk, sim))
    data = data_from_timesteps(steps)
    data['excluded'] = dict(excluded)
    data['removed'] = removed
    return data


def load_inputs(run_dir, gt_h5, vis_npz=None, score_dir=None):
    """채점 입력 모으기. run_dir: frontend_output.h5 폴더 · score_dir: score.json · score_observations.csv · score_kf_gt.csv 폴더
    (기본 run_dir). vis_npz 는 읽지 않는다 (present = 라벨 기준, 26/09/17 결정 ④ — 인자는 호환용)."""
    run_dir = Path(run_dir); score_dir = Path(score_dir or run_dir)
    with h5py.File(run_dir / 'frontend_output.h5', 'r') as f:
        kf_frame = f['kf/frame_idx'][:]
    with h5py.File(gt_h5, 'r') as g:
        ix = g['index']
        ep_oid = ix['gt_object_id'][:]; ep_tid = ix['tracklet_id'][:]
        human_obj = set()
        for e, t in enumerate(ep_tid):
            c = g[f'tracklets/{int(t):05d}/_metadata/color_hex'][()]
            if str(c.decode() if isinstance(c, bytes) else c).lstrip('#') in HUMAN_COLORS:
                human_obj.add(int(ep_oid[e]))
    sj = score_dir / 'score.json'
    p = json.loads(sj.read_text())['params'] if sj.exists() else {}
    tau = float(p.get('tau_m', 0.2))
    threshold = float(p.get('iou_threshold', p.get('min_frac', M.IOU_THRESHOLD)))
    T = int(round(tau * 100))                                               # score_frontend.tau_cm 과 같은 표기
    iou_col = p.get('iou_column', f'gt_iou@{T}'); vox_col = p.get('vox_column', f'gt_vox@{T}')
    kf_gt_file = p.get('kf_gt_file', 'score_kf_gt.csv')
    path = score_dir / 'score_observations.csv'
    rows = list(csv.DictReader(open(path))) if path.exists() and path.stat().st_size else []
    if rows and (iou_col not in rows[0] or vox_col not in rows[0]):
        raise SystemExit(f'{path} 에 {iou_col} · {vox_col} 열이 없다 — 26/09/17 결정 ④ 이후 score_frontend.py(--labels)로 다시 채점할 것')
    kpath = score_dir / kf_gt_file
    if not kpath.exists():
        raise SystemExit(f'{kpath} 없음 — 26/09/17 결정 ④ 이후 score_frontend.py(--labels)로 다시 채점할 것')
    kf_gt = list(csv.DictReader(open(kpath))) if kpath.stat().st_size else []
    return dict(run_dir=run_dir, score_dir=score_dir, n_kf=len(kf_frame), kf_of={int(fr): k for k, fr in enumerate(kf_frame)},
                ep_oid=ep_oid, human_obj=human_obj, rows=rows, kf_gt=kf_gt, tau=tau, min_frac=threshold, threshold=threshold,
                iou_col=iou_col, vox_col=vox_col, kf_gt_file=kf_gt_file, counts_col=p.get('counts_column', f'gt_counts@{T}'),
                score_params=p)


def sequence_data(inp, unit, area_min_px=1600):
    """load_inputs 결과 → TrackEval 입력 dict (unit = 'episode' | 'object')."""
    return build_data(inp['rows'], inp['kf_gt'], inp['n_kf'], inp['ep_oid'], inp['human_obj'], area_min_px, unit,
                      inp['iou_col'], inp['vox_col'], inp['threshold'])


def score_sequence(run_dir, gt_h5, vis_npz=None, area_min_px=1600, out_dir=None, score_dir=None):
    """run_dir: frontend_output.h5 가 있는 폴더. score_dir: score.json · score_observations.csv · score_kf_gt.csv 가 있는 폴더
    (기본 run_dir). out_dir: score_mot.json · score_mot_timeline.json 을 쓸 폴더 (기본 score_dir).
    vis_npz: 호환용 인자 — present 는 score_kf_gt.csv 의 라벨 px 로 정한다."""
    t0 = time.time()
    inp = load_inputs(run_dir, gt_h5, vis_npz, score_dir)
    run_dir, score_dir = inp['run_dir'], inp['score_dir']; out_dir = Path(out_dir or score_dir)
    tau, thr = inp['tau'], inp['threshold']
    out_m, out_tl = {}, {}
    for unit, key in (('episode', 'metrics'), ('object', 'metrics_object')):
        data = sequence_data(inp, unit, area_min_px)
        m, tl = mot_metrics(data, thr)
        m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
        m['n_obs_excluded'] = data['excluded']
        out_m[key] = m; out_tl[unit] = tl
    sp = inp['score_params']
    out = dict(run=str(run_dir), params=dict(
        unit='keyframe', gt_tracks='metrics = 정적 물체의 가시 구간(에피소드) · metrics_object = 시퀀스 전체 물체 (사람 제외)',
        present=f'라벨 px(5m 이내·depth 유효) ≥ {area_min_px} at that keyframe ({inp["kf_gt_file"]} px_label_5m) — 26/09/17 이전엔 gt_vis crop px',
        match=f'CLEAR per keyframe: 1000·[prev-timestep id] + IoU_tau, IoU_tau ≥ {thr} (TrackEval clear.py)',
        fp=f'매칭 안 된 예측 검출 전부 (중복 포함) · 예측 검출 = 무시 GT(present 아님·사람)에 매칭되지 않고 무시 비율 ≤ 0.5 인 관측 '
           f'(TrackEval mot_challenge_2d_box.py distractor 제거 + kitti_mots.py ignore region)',
        idsw='MOT16', hota=f'HOTA_alpha, single alpha = {thr}, TrackEval hota.py matching (global alignment × IoU_tau)',
        tau_m=tau, min_frac=thr, counts_column=inp['counts_col'],
        similarity='tolerant IoU_tau vs per-keyframe GT label surface (score_frontend gt_iou column) — 26/09/17 이전엔 share = gt_counts[oid] / n_vox',
        idf1=f'Ristani 2016 / TrackEval identity.py (potential matches: IoU_tau ≥ {thr})',
        frag='TrackEval clear.py:102-107 (없음 keyframe 도 끊김)', mt='MOT16 ≥ 80% (TrackEval > 0.8 과 MT_ratio_eq_80 만큼 차이)',
        reference='TrackEval commit 12c8791',
        # --- 26/09/17 결정 ④ 추가 ---
        iou_threshold=thr, iou_column=inp['iou_col'], vox_column=inp['vox_col'], kf_gt_file=inp['kf_gt_file'],
        preprocessing='per keyframe match3d.match over all GTs with label surface (same as score_frontend); remove preds matched to '
                      'ignored GTs (non-present or human); remove unmatched preds with (void + voxels nearest to ignored GT)/|P| > 0.5',
        vis_npz_used=False, labels_version=sp.get('labels_version'), labels_meta_sha1=sp.get('labels_meta_sha1')),
        metrics=out_m['metrics'], metrics_object=out_m['metrics_object'], score_dir=str(score_dir),
        seconds=round(time.time() - t0, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'score_mot.json').write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False))
    (out_dir / 'score_mot_timeline.json').write_text(json.dumps(
        {unit: {str(k): [[kk, p] for kk, p in v] for k, v in tl.items()} for unit, tl in out_tl.items()},
        separators=(',', ':')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--gt', required=True)
    ap.add_argument('--vis', default=None, help='호환용 (present 는 score_kf_gt.csv 라벨 기준)')
    ap.add_argument('--score-dir', default=None, help='score.json · score_observations.csv 폴더 (기본 --run)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    r = score_sequence(a.run, a.gt, a.vis, out_dir=a.out, score_dir=a.score_dir)
    print(json.dumps(dict(episode=r['metrics'], object=r['metrics_object']), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
