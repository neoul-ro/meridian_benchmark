#!/usr/bin/env python3
"""score_2d.py 자체 검증 — 손으로 계산한 합성 keyframe.

A. 기본 사례 (처음부터 있던 것)
GT (192x256 격자, 0 = 무시)
  e1 20x20=400칸   e2 20x20=400칸   e3 30x30=900칸   e4 10x10=100칸(작음, 채점 제외)   e5 40x40=1600칸
예측
  p0 = e1 그대로                      → tp, IoU 1
  p1 = e2 ∪ e3 한 덩어리               → e3 와 IoU 900/1300=0.692 로 tp, e2 는 merged, p1 은 merger
  p2 = 무시 영역 덩어리               → ignore
  p3·p4·p5 = e5 를 가로로 3조각       → 각 IoU 0.325·0.325·0.35 → e5 는 split(조각 합집합이 e5 를 100% 덮음), 조각들은 fp
  p6 = e4 그대로                      → tp_small (세지 않음)
기대
  재현율 @0.25 = 3/4 (e5 가 p5 와 0.35) · @0.5 = 2/4 · @0.75 = 1/4 (e3 0.692 탈락)
  TP 2 · FP 3 · FN 2 → precision 2/5 · SQ (1+0.692)/2 · RQ 2/(2+1.5+1) · PQ = SQ×RQ
  (이 사례는 void 겹침이 없고 경계값도 없어 표준 정의로 바꿔도 기대값이 같다)

B. 감사 B 사례 (_audit/B_2d_geometry/synthetic_cases.py 에서 옮김) — 표준 = panopticapi evaluation.py (refs/pq_compute.py)
  사례1  임계 없는 헝가리안이 IoU>0.5 쌍을 버림            (결함 6)
  사례2  void 칸이 IoU 합집합에 들어감                      (결함 1)
  사례3  ignore 경계 void 비율 정확히 0.5 → FP              (결함 10)
  사례4  사람 요약 FP 가 구조적으로 0 — 끝단(파일 입출력)   (결함 2)
  θ<0.5 할당: TP 수 최대 (StarDist matching.py:177)        (결함 6)
  split 은 조각 합집합이 GT 50% 이상 덮을 때만              (결함 8)
  매칭 IoU 정확히 θ → 매칭 아님 (strict >)                  (결함 10)
  mean_best_iou_fn 새 키                                     (결함 14)
  [E3] Loc 경계: 최고 IoU 정확히 0.25 → low_iou (TIDE quantify.py:237 'bg_thresh <= iou'), 경계값은 tide_rules.LOC_MIN_IOU 한 곳
  [F2] Loc 하한 = TIDE 기본 0.1 (quantify.py:428): 정확히 0.1 → low_iou · 0.0975 → missed
"""
import csv
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 채점 코드는 한 칸 위 (frontend_benchmark/)
import score_2d as S  # noqa: E402

FAILS = []


def check(name, got, want, tol=None):
    try:
        ok = abs(got - want) <= tol if tol is not None else got == want
    except TypeError:
        ok = False
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got} want={want}')
    if not ok:
        FAILS.append(name)


def case(fn):
    """사례 하나가 예외로 죽어도 나머지는 돈다 (옛 코드에서 새 키·새 인자가 없을 때 FAIL 로 센다)."""
    print(f'[{fn.__name__}] {fn.__doc__.strip().splitlines()[0]}')
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(limit=2)
        print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
        FAILS.append(fn.__name__)


def box(*boxes):
    x = np.zeros((S.GH, S.GW), bool)
    for r0, r1, c0, c1 in boxes:
        x[r0:r1, c0:c1] = True
    return x


def test_basic():
    """A. 기본 사례 — 상태·재현율·PQ·난이도 누적"""
    G = np.zeros((S.GH, S.GW), np.int32)
    G[10:30, 10:30] = 1; G[10:30, 30:50] = 2; G[40:70, 10:40] = 3; G[100:110, 10:20] = 4; G[100:140, 100:140] = 5
    M = np.stack([box((10, 30, 10, 30)), box((10, 30, 30, 50), (40, 70, 10, 40)), box((150, 180, 150, 200)),
                  box((100, 113, 100, 140)), box((113, 126, 100, 140)), box((126, 140, 100, 140)), box((100, 110, 10, 20))])
    r = S.score_kf(G, M)
    gt = {row['ep_index'] + 1: row for row in r['gt']}
    pr = {row['pred']: row for row in r['pred']}
    print('GT 상태')
    for e, want in [(1, 'tp'), (2, 'merged'), (3, 'tp'), (5, 'split')]:
        check(f'e{e}', gt[e]['status'], want)
    check('e4 는 작아서 채점 제외', gt[4]['eligible'], False)
    check('e3 IoU', gt[3]['best_iou'], 0.6923, 1e-4)
    print('예측 상태')
    for k, want in [(0, 'tp'), (1, 'tp'), (2, 'ignore'), (3, 'fp'), (4, 'fp'), (5, 'fp'), (6, 'tp_small')]:
        check(f'p{k}', pr[k]['status'], want)
    check('p1 은 GT 두 개를 합친 예측', pr[1]['merger'], True)
    check('p0 는 합친 예측 아님', pr[0]['merger'], False)
    print('요약')
    s = S.summarize(r['gt'], r['pred'])
    check('채점 대상 GT 수', s['n_gt_instances'], 4)
    check('재현율 @0.25', s['recall']['0.25'], 0.75)
    check('재현율 @0.5', s['recall']['0.5'], 0.5)
    check('재현율 @0.75', s['recall']['0.75'], 0.25)
    check('정밀도 (TP2 / TP2+FP3)', s['precision'], 0.4)
    sq = (1.0 + 900 / 1300) / 2; rq = 2 / (2 + 1.5 + 1)
    check('SQ', s['SQ'], sq, 1e-3); check('RQ', s['RQ'], rq, 1e-3); check('PQ', s['PQ'], sq * rq, 1e-3)
    check('GT 상태 집계', s['gt_status'], {'tp': 2, 'merged': 1, 'split': 1})
    print('난이도별 (누적)')
    rows = r['gt']
    lv = {1: 0, 2: 1, 3: 2, 4: 0, 5: 3}                       # e1 easy · e2 moderate · e3 hard · e4 작음 · e5 beyond
    for row in rows:
        row['level'] = lv[row['ep_index'] + 1]
    b = S.by_level(rows)
    check('easy 대상 = e1', b['easy']['n'], 1); check('easy 재현율 @0.5 (e1 tp)', b['easy']['recall']['0.5'], 1.0)
    check('moderate 누적 = e1,e2', b['moderate']['n'], 2); check('moderate 재현율 @0.5', b['moderate']['recall']['0.5'], 0.5)
    check('hard 누적 = e1,e2,e3', b['hard']['n'], 3); check('hard 재현율 @0.5 (e1,e3)', b['hard']['recall']['0.5'], 0.6667, 1e-4)
    check('beyond(e5) 는 어느 등급에도 없음', b['hard']['gt_status'].get('split', 0), 0)
    print('빈 경우')
    e = S.score_kf(np.zeros((S.GH, S.GW), np.int32), np.zeros((0, S.GH, S.GW), bool))
    check('GT·예측 없음 → 행 0', (len(e['gt']), len(e['pred'])), (0, 0))
    e2 = S.score_kf(G, np.zeros((0, S.GH, S.GW), bool))
    check('예측 없음 → 채점 대상 전부 missed', S.summarize(e2['gt'], e2['pred'])['gt_status'], {'missed': 4})


def test_case1_thresholded_hungarian():
    """사례1: P=g1∪g2 (IoU 0.6/0.4), Q⊂g1 (IoU 0.45) → 표준 TP1 FP1 FN1, PQ 0.3 (결함 6)"""
    G = np.zeros((S.GH, S.GW), np.int32)
    G[10:40, 10:30] = 1; G[10:30, 30:50] = 2
    M = np.stack([box((10, 40, 10, 30), (10, 30, 30, 50)), box((10, 37, 10, 20))])
    r = S.score_kf(G, M); s = S.summarize(r['gt'], r['pred'])
    gt = {row['ep_index'] + 1: row for row in r['gt']}
    check('g1 tp (P 와 IoU 0.6)', gt[1]['status'], 'tp')
    check('P 는 tp', r['pred'][0]['status'], 'tp'); check('Q 는 fp', r['pred'][1]['status'], 'fp')
    check('재현율 @0.5', s['recall']['0.5'], 0.5)
    check('SQ (panopticapi)', s['SQ'], 0.6, 1e-4); check('RQ', s['RQ'], 0.5, 1e-4); check('PQ', s['PQ'], 0.3, 1e-4)


def test_case2_void_excluded_from_union():
    """사례2: GT 400칸 + void 500칸을 덮은 예측 → IoU 400/(900+400-400-500) = 1 (결함 1, Kirillov §4.1)"""
    G = np.zeros((S.GH, S.GW), np.int32); G[50:70, 50:70] = 1
    M = np.stack([box((50, 70, 50, 95))])
    r = S.score_kf(G, M); s = S.summarize(r['gt'], r['pred'])
    check('IoU = 1', r['gt'][0]['best_iou'], 1.0, 1e-6)
    check('GT tp', r['gt'][0]['status'], 'tp'); check('예측 tp', r['pred'][0]['status'], 'tp')
    check('PQ = 1', s['PQ'], 1.0, 1e-6)


def test_case3_ignore_strict():
    """사례3: 예측 600칸 중 void 300칸(정확히 0.5) → ignore 아님 = FP (결함 10, evaluation.py:161 '> 0.5')"""
    G = np.zeros((S.GH, S.GW), np.int32); G[100:140, 100:125] = 1
    M = np.stack([box((100, 130, 125, 135), (100, 130, 100, 110))])
    r = S.score_kf(G, M); s = S.summarize(r['gt'], r['pred'])
    check('void 비율 0.5', r['pred'][0]['ignore_frac'], 0.5, 1e-6)
    check('예측 fp', r['pred'][0]['status'], 'fp')
    check('IoU (void 제외) = 300/1000', r['gt'][0]['best_iou'], 0.3, 1e-6)
    check('TP0 FP1 FN1 → RQ 0', (s['pred_status'].get('fp', 0), s['RQ']), (1, 0.0))


def test_theta_low_cardinality():
    """θ=0.25: A-g1 0.7 한 쌍보다 A-g2 0.26 + B-g1 0.3 두 쌍 — TP 수 최대 (StarDist matching.py:177, 결함 6)"""
    G = np.zeros((S.GH, S.GW), np.int32)
    G[0:35, 0:20] = 1          # g1 700칸
    G[0:22, 20:40] = 2         # g2 440칸
    M = np.stack([box((0, 35, 0, 20), (0, 15, 20, 40)),     # A = g1 + g2 의 300칸 → IoU g1 0.7, g2 300/1140
                  box((0, 21, 0, 10))])                     # B ⊂ g1 210칸 → IoU 0.3
    r = S.score_kf(G, M); s = S.summarize(r['gt'], r['pred'])
    check('재현율 @0.25 = 2/2', s['recall']['0.25'], 1.0)
    check('재현율 @0.5 = 1/2 (A-g1)', s['recall']['0.5'], 0.5)
    check('θ=0.5 매칭 = A-g1', r['matches'][0.5], [(0, 0)])


def test_split_coverage():
    """split: 조각 합집합이 GT 의 50% 이상 덮어야 (결함 8)"""
    G = np.zeros((S.GH, S.GW), np.int32); G[100:140, 100:140] = 1          # 1600칸
    M = np.stack([box((100, 108, 100, 140)), box((120, 128, 100, 140))])  # 각 20%, 합집합 40%
    r = S.score_kf(G, M)
    check('합집합 40% → split 아님 (best IoU 0.2 ≥ 0.1 → low_iou, F2)', r['gt'][0]['status'], 'low_iou')
    M = np.stack([box((100, 115, 100, 140)), box((102, 117, 100, 140))])  # 각 37.5%, 합 75%, 합집합 42.5%
    r = S.score_kf(G, M)
    check('겹친 조각: 합 75% 이지만 합집합 42.5% → split 아님 (IoU 0.375 → low_iou)', r['gt'][0]['status'], 'low_iou')
    M = np.stack([box((100, 114, 100, 140)), box((113, 127, 100, 140)), box((126, 140, 100, 140))])  # 합집합 100%, 각 IoU 0.35
    r = S.score_kf(G, M)
    check('합집합 100% → split', r['gt'][0]['status'], 'split')


def test_match_strict():
    """IoU 가 정확히 0.5 → @0.5 매칭 아님 (evaluation.py:134 'iou > 0.5', 결함 10)"""
    G = np.zeros((S.GH, S.GW), np.int32); G[0:20, 200:220] = 1           # 400칸
    M = np.stack([box((0, 10, 200, 220))])                               # 200칸 ⊂ GT → IoU 0.5
    r = S.score_kf(G, M)
    check('IoU 0.5', r['gt'][0]['best_iou'], 0.5, 1e-9)
    check('tp@0.5 = 0', r['gt'][0]['tp@0.5'], 0)
    check('tp@0.25 = 1', r['gt'][0]['tp@0.25'], 1)
    check('상태 low_iou', r['gt'][0]['status'], 'low_iou')


def test_mean_best_iou_fn():
    """mean_best_iou_fn = FN(매칭 안 된 채점 대상 GT 전부)의 최고 IoU 평균 (결함 14)"""
    G = np.zeros((S.GH, S.GW), np.int32); G[0:20, 0:20] = 1; G[50:70, 50:70] = 2; G[100:120, 100:120] = 3
    M = np.stack([box((0, 20, 0, 20)), box((50, 60, 50, 70))])           # g1 tp, g2 IoU 0.5 (low_iou), g3 missed IoU 0
    r = S.score_kf(G, M); s = S.summarize(r['gt'], r['pred'])
    check('mean_best_iou_fn = (0.5+0)/2', s.get('mean_best_iou_fn'), 0.25, 1e-6)
    check('옛 키 mean_best_iou_missed 유지 (deprecated, 같은 값)', s.get('mean_best_iou_missed'), 0.25, 1e-6)


def test_loc_boundary_shared():
    """[E3] Loc 경계 = tide_rules.LOC_MIN_IOU 이상 (TIDE quantify.py:237 'bg <= IoU') — 3D match3d 와 같은 상수·같은 비교"""
    import tide_rules as TR
    G = np.zeros((S.GH, S.GW), np.int32); G[0:20, 0:20] = 1                 # 400칸
    M = np.stack([box((0, 5, 0, 20))])                                    # 100칸 ⊂ GT → IoU 정확히 0.25
    r = S.score_kf(G, M)
    check('IoU 0.25', r['gt'][0]['best_iou'], 0.25, 1e-12)
    check('정확히 0.25 → low_iou (옛 코드: > 0.25 라 missed)', r['gt'][0]['status'], 'low_iou')
    check('2D 도 같은 함수: is_loc(0.25, θ) 참 · is_loc(θ 초과) 거짓', (TR.is_loc(0.25, 0.5), TR.is_loc(0.51, 0.5)), (True, False))
    r = S.score_kf(G, np.stack([box((0, 2, 0, 20))]))                     # 40칸 ⊂ GT → IoU 정확히 0.1
    check('[F2] IoU 정확히 0.1 → low_iou (TIDE bg 기본 0.1 포함)', (r['gt'][0]['best_iou'], r['gt'][0]['status']), (0.1, 'low_iou'))
    r = S.score_kf(G, np.stack([box((0, 2, 0, 19)) | box((0, 1, 0, 1))]))  # 38칸 → IoU 0.095
    check('[F2] IoU 0.095 → missed', (r['gt'][0]['best_iou'], r['gt'][0]['status']), (0.095, 'missed'))
    old = TR.LOC_MIN_IOU
    try:
        TR.LOC_MIN_IOU = 0.3                                              # 한 곳만 바꾸면 2D 판정도 따라 바뀐다
        check('LOC_MIN_IOU 0.3 → missed (score_2d 가 tide_rules 값을 읽음)', S.score_kf(G, M)['gt'][0]['status'], 'missed')
    finally:
        TR.LOC_MIN_IOU = old


# ------------------------------------------------------------------ 끝단: 파일 입출력 · 사람/정적 분리 · 인터페이스
OLD_JSON_KEYS = ['run', 'labels', 'params', 'all', 'static', 'human', 'recall_by_size', 'by_level', 'seconds']
OLD_SUMMARY_KEYS = ['n_gt_instances', 'n_pred_masks', 'recall', 'precision', 'SQ', 'RQ', 'PQ', 'gt_status', 'pred_status',
                    'mean_best_iou_missed']
OLD_GT_COLS = ('ep_index,area_cells,eligible,status,best_iou,matched_pred,best_pred,tp@0.25,tp@0.5,tp@0.75,trunc,occ,min_dim,'
               'level,kf,frame,gt_object_id,is_human,matched_obs,best_obs').split(',')
OLD_PRED_COLS = 'pred,area_cells,ignore_frac,status,merger,matched_ep,iou,kf,frame,obs,tracklet_id'.split(',')


def cells_to_png(Gc):
    """192x256 격자 라벨 → 640x480 픽셀 라벨 (gt_grid 가 칸 중심 픽셀로 되읽으면 그대로 나오게)."""
    v = np.minimum((np.arange(480) / S.CELL).astype(int), S.GH - 1)
    u = np.minimum((np.arange(640) / S.CELL).astype(int), S.GW - 1)
    return Gc[v][:, u].astype(np.uint16)


def write_fake_run(root, Gc, masks, colors, oids):
    import cv2
    import h5py
    run = root / 'run'; lab = root / 'labels'; run.mkdir(); lab.mkdir()
    bits = np.packbits(masks.reshape(len(masks), -1), axis=1)
    with h5py.File(run / 'frontend_output.h5', 'w') as f:
        f['kf/frame_idx'] = np.array([0]); f['kf/obs_start'] = np.array([0]); f['kf/n_obs'] = np.array([len(masks)])
        f['obs/tracklet_id'] = np.arange(len(masks)); f['obs/mask_bits'] = bits
    with h5py.File(root / 'gt.h5', 'w') as g:
        g['index/gt_object_id'] = np.array(oids); g['index/tracklet_id'] = np.arange(len(oids))
        for t, c in enumerate(colors):
            g[f'tracklets/{t:05d}/_metadata/color_hex'] = np.bytes_('#' + c)
    cv2.imwrite(str(lab / '000000.png'), cells_to_png(Gc))
    return run, root / 'gt.h5', lab


def test_end_to_end_human_static():
    """끝단: 사람 GT 의 중복 조각 예측이 사람 요약 FP 로 잡히고, 기존 JSON 키·CSV 열이 그대로 (결함 2 · 인터페이스)"""
    Gc = np.zeros((S.GH, S.GW), np.int32)
    Gc[20:60, 20:40] = 1          # ep0 사람 800칸
    Gc[100:140, 100:140] = 2      # ep1 정적 1600칸
    M = np.stack([box((20, 60, 20, 40)), box((20, 45, 20, 40)),          # 사람 그대로 · 사람 조각(중복)
                  box((100, 140, 100, 140)), box((100, 110, 100, 110))])  # 정적 그대로 · 정적 조각
    with tempfile.TemporaryDirectory() as d:
        run, gt, lab = write_fake_run(Path(d), Gc, M, ['23d5ea', 'aabbcc'], [7, 8])
        out = S.score_sequence(run, gt, lab, out_dir=Path(d) / 'out', log=lambda *_: None)
        hs, ss, al = out['human'], out['static'], out['all']
        check('사람: TP1 FP1 → 정밀도 0.5', hs['precision'], 0.5)
        check('사람: RQ = 1/(1+0.5)', hs['RQ'], round(1 / 1.5, 4))
        check('사람 예측 상태', hs['pred_status'], {'tp': 1, 'fp': 1})
        check('정적: TP1 FP1 → 정밀도 0.5', ss['precision'], 0.5)
        check('정적 예측 상태', ss['pred_status'], {'tp': 1, 'fp': 1})
        check('전체 정밀도 2/4', al['precision'], 0.5)
        check('사람+정적 예측 수 = 전체', hs['n_pred_masks'] + ss['n_pred_masks'], al['n_pred_masks'])
        j = json.loads((Path(d) / 'out' / 'score_2d.json').read_text())
        check('JSON 기존 최상위 키 유지', [k for k in OLD_JSON_KEYS if k not in j], [])
        check('[F2] params.loc_min_iou = tide_rules.LOC_MIN_IOU (TIDE 기본 0.1)', j['params'].get('loc_min_iou'), 0.1)
        for grp in ('all', 'static', 'human'):
            check(f'JSON {grp} 기존 키 유지', [k for k in OLD_SUMMARY_KEYS if k not in j[grp]], [])
        gcols = next(csv.reader(open(Path(d) / 'out' / 'score_2d_gt.csv')))
        pcols = next(csv.reader(open(Path(d) / 'out' / 'score_2d_pred.csv')))
        check('GT CSV 기존 열 유지 (difficulty 없음 → trunc 등 4열 제외)',
              [c for c in OLD_GT_COLS if c not in gcols and c not in ('trunc', 'occ', 'min_dim', 'level')], [])
        check('pred CSV 기존 열 유지', [c for c in OLD_PRED_COLS if c not in pcols], [])
        check('pred CSV 기존 열 위치 그대로 (새 열은 뒤)', pcols[:len(OLD_PRED_COLS)], OLD_PRED_COLS)
        check('GT CSV 기존 열 순서 그대로', [c for c in gcols if c in OLD_GT_COLS], [c for c in OLD_GT_COLS if c in gcols])
        check('GT CSV 새 열은 뒤', gcols[-1], 'split_cover')


def run():
    for fn in (test_basic, test_case1_thresholded_hungarian, test_case2_void_excluded_from_union, test_case3_ignore_strict,
               test_theta_low_cardinality, test_split_coverage, test_match_strict, test_mean_best_iou_fn,
               test_loc_boundary_shared, test_end_to_end_human_static):
        case(fn)
    return FAILS


def main():
    # [E4] 자기 사례만 돈다. 예전 main 은 test_gt_labels_2d · test_gt_difficulty_2d 의 run() 까지 불러, 직접 실행하면 두 파일을
    # run_tests.py(eval.sh test) 와 겹쳐 두 번 돌렸다. 두 파일은 각자 실행한다.
    run()
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
