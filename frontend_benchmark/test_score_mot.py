#!/usr/bin/env python3
"""score_mot.py 자체 검증 — 손으로 계산한 합성 트랙 (표준 정의: TrackEval clear.py·identity.py·hota.py, MOT16, Ristani 2016).

유사도 (26/09/17 결정 ④): similarity[g, 예측] = score_frontend 의 허용 거리 IoU_τ (score_observations.csv 의 gt_iou@<τ>).
GT 검출 = 그 keyframe 에 present(라벨 5m 이내 ≥ 1600px, score_kf_gt.csv)인 정적 GT. 사람·present 아닌 GT 는 무시 GT.

1. 지표 알고리즘 (data_from_timesteps 로 timestep 별 GT·예측·similarity 를 직접 준다)
   GT 물체 A·B 가 keyframe 0~9 에 모두 있음 (GT 검출 20), similarity 는 매칭 관측 1.0
     A: k0~4 id1 · k5 검출 없음 · k6~9 id2              → TP 9 · FN 1 · IDSW 1 · Frag 1 · MT (90%)
     B: k0~9 id3 · k3 에 id4 중복 관측 하나(IoU 0.6)     → TP 10 · FP 1 · MT
   기대
     MOTA = 1 − (1 + 1 + 1)/20 = 0.85
     IDF1: 공동 출현 A-1 5 · A-2 4 · B-3 10 · B-4 1 → 헝가리안 A-1, B-3 → IDTP 15 · IDFN 5 · IDFP 20−15 = 5 → 30/40 = 0.75
     HOTA: global alignment B-3 ≫ B-4 라 k3 도 id3 → DetA = 19/21 · AssA = (5·5/10 + 4·4/10 + 10·10/10)/19 = 14.1/19
2. 전처리 (build_data): score_frontend 와 같은 keyframe 매칭(match3d.match)으로
     · present 아닌 GT·사람 GT 에 매칭된 예측 → 제거 (TrackEval mot_challenge_2d_box.py:370-381 distractor 매칭 제거)
     · 매칭 안 된 예측 중 (void + 무시 GT 위 복셀) > 50% → 제거 (kitti_mots.py:336-344 · panopticapi evaluation.py:161)
     · 복셀 0 → 제거. 나머지 = 예측 검출
3. 끝에서 끝까지 (score_frontend → score_mot): 감사 A 결함 재현 사례 + IoU 유사도 사례
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_mot as S  # noqa: E402

FAILS = []


def check(name, got, want, tol=None):
    try:
        ok = (got is not None and abs(got - want) <= tol) if tol is not None else got == want
    except TypeError:
        ok = False
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got} want={want}')
    if not ok:
        FAILS.append(name)


def steps_from(present, dets, n):
    """present: {k: [GT 키]} · dets: {k: [(예측 키, {GT 키: similarity})]} → data."""
    out = []
    for k in range(n):
        gk = sorted(present.get(k, [])); dk = dets.get(k, [])
        sim = np.array([[d.get(g, 0.0) for _, d in dk] for g in gk]).reshape(len(gk), len(dk))
        out.append((gk, [p for p, _ in dk], sim))
    return S.data_from_timesteps(out)


def main():
    A, B = 1, 2
    print('기본')
    present = {k: [A, B] for k in range(10)}
    dets = {k: [(1 if k < 5 else 2, {A: 1.0})] if k != 5 else [] for k in range(10)}
    for k in range(10):
        dets[k] = dets[k] + [(3, {B: 1.0})] + ([(4, {B: 0.6})] if k == 3 else [])
    m, tl = S.mot_metrics(steps_from(present, dets, 10))
    check('GT 검출', m['gt_detections'], 20); check('TP', m['TP'], 19); check('FN', m['FN'], 1); check('FP', m['FP'], 1)
    check('IDSW', m['IDSW'], 1); check('Frag', m['Frag'], 1); check('MT', m['MT'], 2); check('ML', m['ML'], 0)
    check('MOTA', m['MOTA'], 0.85, 1e-9)
    check('IDTP', m['IDTP'], 15); check('IDFN', m['IDFN'], 5); check('IDFP', m['IDFP'], 5); check('IDF1', m['IDF1'], 0.75, 1e-9)
    det = 19 / 21; ass = (5 * 0.5 + 4 * 0.4 + 10 * 1.0) / 19
    check('DetA', m['DetA'], det, 1e-9); check('AssA', m['AssA'], ass, 1e-9); check('HOTA', m['HOTA'], (det * ass) ** 0.5, 1e-9)
    check('HOTA_TP·FN·FP', (m['HOTA_TP'], m['HOTA_FN'], m['HOTA_FP']), (19, 1, 1))
    check('A 타임라인 k5 놓침', tl[A][5], (5, None))
    check('A 타임라인 k6 id2', tl[A][6], (6, 2))

    print('ID 가 1→2→1 로 돌아와도 매번 스위치 (MOT16)')
    m2, _ = S.mot_metrics(steps_from({k: [A] for k in range(3)}, {0: [(1, {A: 1.0})], 1: [(2, {A: 1.0})], 2: [(1, {A: 1.0})]}, 3))
    check('IDSW 2', m2['IDSW'], 2); check('Frag 0 (끊김 없음)', m2['Frag'], 0)

    print('놓친 구간 세 번 (그 keyframe 에 다른 예측 검출이 있어 연속성이 초기화됨)')
    # 옛 테스트는 놓친 keyframe 에 예측 검출이 아예 없었고 Frag 3 을 기대했다. TrackEval clear.py:73-76 은 예측 검출이 0개인
    # timestep 에서 연속성 기억을 지우지 않으므로 그 입력의 표준 값은 Frag 0 이다 (아래 '검출 없는 keyframe' 에서 따로 확인).
    d3 = {k: [(1, {A: 1.0})] if k % 2 == 0 else [(9, {A: 0.3})] for k in range(7)}
    m3, _ = S.mot_metrics(steps_from({k: [A] for k in range(7)}, d3, 7))
    check('Frag 3', m3['Frag'], 3); check('IDSW 0', m3['IDSW'], 0); check('PT (4/7)', (m3['MT'], m3['PT'], m3['ML']), (0, 1, 0))
    check('FP 3 (id9)', m3['FP'], 3)
    print('검출 없는 keyframe 은 끊김이 아니다 (TrackEval clear.py:73-76 공식 구현 동작)')
    m3b, _ = S.mot_metrics(steps_from({k: [A] for k in range(7)}, {k: [(1, {A: 1.0})] for k in range(0, 7, 2)}, 7))
    check('Frag 0', m3b['Frag'], 0); check('FN 3', m3b['FN'], 3)

    print('아무것도 못 잡음')
    m4, _ = S.mot_metrics(steps_from({k: [A] for k in range(5)}, {}, 5))
    # HOTA 식 13 = √(Σ_TP A(c) / (TP+FN+FP)) = √(0/5) = 0. AssA 는 평균할 TP 가 없어 null.
    check('MOTA 0', m4['MOTA'], 0.0, 1e-9); check('ML 1', m4['ML'], 1)
    check('HOTA 0 · DetA 0 · AssA null', (m4['HOTA'], m4['DetA'], m4['AssA']), (0.0, 0.0, None))
    check('IDF1 0', m4['IDF1'], 0.0, 1e-9)

    print('MT 경계 — 정확히 80% 는 MT (MOT16), 20% 는 PT')
    m5, _ = S.mot_metrics(steps_from({k: [A, B] for k in range(5)},
                                      {k: ([(1, {A: 1.0})] if k < 4 else []) + ([(2, {B: 1.0})] if k < 1 else []) for k in range(5)}, 5))
    check('MT/PT/ML', (m5['MT'], m5['PT'], m5['ML']), (1, 1, 0)); check('MT_ratio_eq_80', m5['MT_ratio_eq_80'], 1)


def test_build_data():
    print('전처리 build_data — 무시 GT(present 아님·사람)에 매칭된 예측 제거 · 무시 비율 > 50% 제거 · 단위')
    # 옛 테스트는 share 합(gt_counts) 과 가시성(vis px) 으로 예측 검출을 골랐다. 새 입력은 IoU(gt_iou) · 에피소드별 가장 가까운
    # 표면 복셀 수(gt_vox) · score_kf_gt.csv 의 라벨 px 다.
    col_i, col_v = 'gt_iou@20', 'gt_vox@20'
    rows = [dict(kf=0, tracklet_id=1, n_vox=100, **{col_i: '0:1.0', col_v: '0:100'}),           # ep0 매칭 → 검출
            dict(kf=0, tracklet_id=2, n_vox=100, **{col_i: '0:0.3;2:0.2', col_v: '0:30;2:30'}),  # 미매칭 · 무시 70% → 제거
            dict(kf=0, tracklet_id=3, n_vox=100, **{col_i: '3:0.9', col_v: '3:100'}),           # 사람 GT 매칭 → 제거
            dict(kf=0, tracklet_id=4, n_vox=0, **{col_i: '', col_v: ''}),                       # 5m 밖 → 제거
            dict(kf=0, tracklet_id=7, n_vox=100, **{col_i: '2:0.8', col_v: '2:100'}),           # present 아닌 ep2 매칭 → 제거
            dict(kf=0, tracklet_id=8, n_vox=100, **{col_i: '0:0.6', col_v: '0:50'}),            # ep0 은 id1 이 가짐 · 무시 50% → FP
            dict(kf=1, tracklet_id=5, n_vox=10, **{col_i: '1:0.5', col_v: '1:5'}),              # IoU 정확히 0.5 → 검출
            dict(kf=2, tracklet_id=6, n_vox=10, **{col_i: '1:1.0', col_v: '1:10'})]
    kfgt = [dict(kf=0, ep_index=0, px_label_5m=5000), dict(kf=0, ep_index=2, px_label_5m=900),
            dict(kf=0, ep_index=3, px_label_5m=5000), dict(kf=1, ep_index=1, px_label_5m=5000),
            dict(kf=2, ep_index=1, px_label_5m=1600)]
    ep_oid = np.array([1, 1, 3, 4])                  # ep0·ep1 = 물체1, ep2 = 물체3, ep3 = 사람(물체4)
    de = S.build_data(rows, kfgt, 3, ep_oid, {4}, 1600, 'episode', col_i, col_v, 0.5)
    check('예측 검출 수 (id1, id8, id5, id6)', de['num_tracker_dets'], 4)
    check('제거 사유', de['excluded'], {'mostly_ignore_region': 1, 'matched_human_gt': 1, 'no_voxels_within_range': 1,
                                    'matched_non_present_gt': 1})
    check('kf1 similarity (ep1 · id5) = 0.5', float(de['similarity_scores'][1][0, 0]), 0.5)
    check('kf0 GT 검출 = ep0 하나 (ep2 900px · ep3 사람 제외)', len(de['gt_ids'][0]), 1)
    me, _ = S.mot_metrics(de)
    do = S.build_data(rows, kfgt, 3, ep_oid, {4}, 1600, 'object', col_i, col_v, 0.5)
    mo, _ = S.mot_metrics(do)
    check('에피소드 단위: GT 트랙 2 · IDSW 1 (ep1 안에서 id5→id6) · FP 1 (id8)', (me['n_gt_tracks'], me['IDSW'], me['FP']), (2, 1, 1))
    check('물체 단위: GT 트랙 1 · IDSW 2 (id1→id5→id6)', (mo['n_gt_tracks'], mo['IDSW']), (1, 2))
    dp = S.build_data(rows, kfgt, 3, ep_oid, {4}, 1601, 'episode', col_i, col_v, 0.5)
    check('area_min_px 1601 → kf2 의 ep1(1600px) 은 present 아님 → id6 제거 사유 matched_non_present_gt 2',
          dp['excluded'].get('matched_non_present_gt'), 2)


# ---------------------------------------------------------------------------------------------------------------
# 끝에서 끝까지(score_frontend → score_mot) 합성 사례. 기대값은 표준 정의(TrackEval clear.py·identity.py·hota.py,
# MOT16, Ristani 2016)로 손계산. 격자 τ=20cm: 같은 열 방향 9셀 이하만 τ 이내.
#   A = 50×50 평면 · part75 = A 앞 30행 + GT 없는 500셀 → P 0.75 · R 0..38행 = 0.78 → IoU = 0.585/0.945 = 13/21
# ---------------------------------------------------------------------------------------------------------------
def _ts():
    import test_score as TS                       # plane · FakeGT · FakeSurface · write_run (같은 폴더의 3D 채점 테스트 도우미)
    return TS


def write_gt_h5(path, gt):
    """score_mot 이 읽는 GT h5 의 최소 레이아웃 (index/gt_object_id · tracklet_id · 색)."""
    with h5py.File(path, 'w') as f:
        f['index/gt_object_id'] = np.asarray(gt.oid, np.int64)
        f['index/tracklet_id'] = np.asarray(gt.tid, np.int64)
        for t, c in zip(gt.tid, gt.color):
            f[f'tracklets/{int(t):05d}/_metadata/color_hex'] = str(c)


def surface_from_vis(gt, vis):
    """옛 테스트의 가시성 입력(에피소드·프레임·px)을 그대로 라벨 표면으로: 그 쌍에만 표면(에피소드 점군), 라벨 px = 그 px."""
    TS = _ts()
    return TS.FakeSurface(gt, pairs={(int(f), int(e)): int(p) for e, f, p in zip(vis['ep_index'], vis['frame'], vis['px_crop'])})


def e2e(tmp, name, gt, kfs, vis, tau=0.2, n_frames=400, surface=None):
    TS = _ts()
    d = tmp / name
    TS.write_run(d, kfs, n_frames)
    TS.S.score(d, TS.SEQ, None, tau=tau, max_range=1e9, gt_obj=gt, out_dir=d, vis=vis, log=lambda *a: None,
               surface=surface if surface is not None else surface_from_vis(gt, vis))
    write_gt_h5(d / 'gt.h5', gt)
    np.savez(d / 'vis.npz', **vis)
    out = S.score_sequence(d, d / 'gt.h5', d / 'vis.npz')
    return out['metrics'], out


def guarded(fn):
    def wrap(*a):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(f'{fn.__name__} 예외')
    return wrap


def vis_all(n_eps, frames, px=5000):
    return dict(ep_index=np.repeat(np.arange(n_eps), len(frames)), frame=np.tile(np.array(frames), n_eps),
                px_crop=np.full(n_eps * len(frames), px))


@guarded
def test_idf1_hota_duplicates(tmp):
    print('E1. [감사 A-1·A-2] 중복 관측의 공동 출현을 IDF1 에 넣고, HOTA 는 global alignment 로 다시 매칭')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); far = TS.plane(40, 40, 0.0, 20, 25)
    part75 = np.concatenate([A[:1500], far])                 # IoU 13/21
    gt = TS.FakeGT([dict(oid=1, first=0, last=200, pts=A)])
    fr = [10 * (i + 1) for i in range(10)]
    kfs = [(f, ([(1, A)] if i < 5 else []) + [(2, part75)]) for i, f in enumerate(fr)]
    m, out = e2e(tmp, 'idf1', gt, kfs, vis_all(1, fr))
    # CLEAR: kf0~4 id1(1.0) 매칭·id2(13/21) FP, kf5~9 id2 → IDSW 1. MOTA = 1-(0+5+1)/10
    check('TP', m['TP'], 10); check('FP', m['FP'], 5); check('IDSW', m['IDSW'], 1); check('MOTA', m['MOTA'], 0.4, 1e-4)
    # Ristani 식 3~8 / identity.py:53-57: id2 는 10 keyframe 모두 IoU ≥ 0.5 → IDTP 10 · IDFP 5 · IDFN 0
    check('IDTP', m['IDTP'], 10); check('IDFP', m['IDFP'], 5); check('IDFN', m['IDFN'], 0); check('IDF1', m['IDF1'], 0.8, 1e-4)
    # hota.py:52-68 (s = 13/21): kf0~4 sim_iou A-1 = 21/34 · A-2 = 13/34, kf5~9 A-2 = 1
    #   potential A-1 = 105/34 · A-2 = 235/34 → GA A-1 = 105/405 = 0.259 · A-2 = 235/445 = 0.528
    #   kf0~4 점수 id1 0.259·1 < id2 0.528·0.619 = 0.327 → 전부 id2 → AssA 1
    check('DetA', m['DetA'], 10 / 15, 1e-4); check('AssA', m['AssA'], 1.0, 1e-4); check('HOTA', m['HOTA'], (10 / 15) ** 0.5, 1e-4)


@guarded
def test_similarity_is_iou(tmp):
    print('E0. [결정 ④] similarity = score_frontend 의 IoU_τ (gt_iou@20) — share(0.75) 가 아니라 13/21')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); far = TS.plane(40, 40, 0.0, 20, 25)
    gt = TS.FakeGT([dict(oid=1, first=0, last=200, pts=A)])
    _, out = e2e(tmp, 'simiou', gt, [(10, [(1, A), (2, np.concatenate([A[:1500], far]))])], vis_all(1, [10]))
    d = tmp / 'simiou'
    inp = S.load_inputs(d, d / 'gt.h5', d / 'vis.npz')
    data = S.sequence_data(inp, 'episode')
    sim = data['similarity_scores'][0]
    check('similarity 모양 (GT 1 × 예측 2)', sim.shape, (1, 2))
    check('similarity[A, id1] = 1', float(sim[0, 0]), 1.0, 1e-12)
    check('similarity[A, id2] = 13/21', float(sim[0, 1]), 13 / 21, 1e-12)
    check('params.similarity 에 IoU', 'IoU' in out['params']['similarity'], True)
    check('params.iou_column', out['params'].get('iou_column'), 'gt_iou@20')


@guarded
def test_fragment_mot(tmp):
    print('E9. [결정 ④] 작은 순수 조각 (P 1 · IoU 0.26) — share 로는 TP 였지만 IoU 로는 FP + FN')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50)
    frag = A[:200]                                           # 4행: R = 0..12행 = 650/2500 = 0.26
    gt = TS.FakeGT([dict(oid=1, first=0, last=200, pts=A)])
    m, _ = e2e(tmp, 'fragmot', gt, [(10, [(1, frag)]), (20, [(1, frag)])], vis_all(1, [10, 20]))
    check('TP 0 · FN 2 · FP 2', (m['TP'], m['FN'], m['FP']), (0, 2, 2))
    check('MOTA = 1 − 4/2 = −1', m['MOTA'], -1.0, 1e-9)
    d = tmp / 'fragmot'
    inp = S.load_inputs(d, d / 'gt.h5', d / 'vis.npz')
    check('similarity = 0.26', float(S.sequence_data(inp, 'episode')['similarity_scores'][0][0, 0]), 0.26, 1e-12)


@guarded
def test_fp_merge(tmp):
    print('E2. [감사 A-3] 있는 GT 두 개를 한 관측으로 합침 — 매칭 안 됐지만 무시 비율 12% → FP · 두 GT 는 merged')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); B = TS.plane(3, 0, 0.0, 50, 50); far = TS.plane(40, 40, 0.0, 20, 25)
    gt = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A), dict(oid=2, first=0, last=100, pts=B)])
    # 각 GT 22행(1100) + void 300: P 0.44 · R 0..30행 = 0.62 → IoU = 0.2728/0.7872 = 0.3465 (< 0.5) · R ≥ 0.5 두 개 → merged
    merged = np.concatenate([A[:1100], B[:1100], far[:300]])
    m, _ = e2e(tmp, 'fpmerge', gt, [(10, [(1, merged)])], vis_all(2, [10]))
    check('TP 0', m['TP'], 0); check('FN 2', m['FN'], 2); check('FP 1', m['FP'], 1)
    check('MOTA = 1-(2+1)/2', m['MOTA'], -0.5, 1e-4)
    kg = _ts().read_csv(tmp / 'fpmerge/score_kf_gt.csv')
    check('score_kf_gt 상태 merged · merged', [r['status'] for r in kg], ['merged', 'merged'])


@guarded
def test_fp_ignore_regions(tmp):
    print('E3. [감사 A-3] FP 에서 빼는 관측 — 무시 비율 과반 · present 아닌 GT 에 매칭 · 사람 GT 에 매칭')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); Bs = TS.plane(3, 0, 0.0, 50, 50); H = TS.plane(0, 3, 0.0, 50, 50)
    far = TS.plane(40, 40, 0.0, 50, 50)
    gt = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A), dict(oid=2, first=0, last=100, pts=Bs),
                    dict(oid=3, first=0, last=100, pts=H, color='23d5ea')])
    vis = dict(ep_index=np.array([0, 1, 2]), frame=np.array([10, 10, 10]), px_crop=np.array([5000, 800, 5000]))
    kfs = [(10, [(1, A),                                          # 있는 GT 매칭 → TP
                 (2, np.concatenate([A[:1000], far[:1500]])),     # IoU 0.31 미매칭 · void 60% → 제외
                 (3, Bs),                                         # present 아닌(800px) GT 에 매칭 → 제외
                 (4, H),                                          # 사람 GT 에 매칭 → 제외
                 (5, np.concatenate([A[:1250], far[:1250]]))])]   # IoU 0.405 미매칭 · void 정확히 50% → FP
    m, out = e2e(tmp, 'fpign', gt, kfs, vis)
    check('GT 검출 1 (사람·작은 GT 제외)', m['gt_detections'], 1)
    check('TP 1', m['TP'], 1); check('FP 1 (id5 만)', m['FP'], 1); check('MOTA 0', m['MOTA'], 0.0, 1e-4)
    check('제거 사유', m['n_obs_excluded'], {'mostly_ignore_region': 1, 'matched_non_present_gt': 1, 'matched_human_gt': 1})


@guarded
def test_continuity_prev_timestep(tmp):
    print('E4. [감사 A-4] 연속성은 바로 이전 timestep 에 매칭된 쌍만 — 놓친 keyframe·present 아닌 keyframe 뒤엔 초기화')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); B = TS.plane(3, 0, 0.0, 50, 50); far = TS.plane(40, 40, 0.0, 20, 25)
    part75 = np.concatenate([A[:1500], far])
    gt = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A), dict(oid=2, first=0, last=100, pts=B)])
    # (a) C1: kf1 에서 A 놓침(id1 이 벽만 보냄 → 제외) → kf2 에서 id1(13/21) 과 id2(1.0) 중 id2 → IDSW 1
    kfs = [(10, [(1, A), (9, B)]), (20, [(1, far), (9, B)]), (30, [(1, part75), (2, A), (9, B)])]
    m, _ = e2e(tmp, 'cont_a', gt, kfs, vis_all(2, [10, 20, 30]))
    check('(a) IDSW 1', m['IDSW'], 1); check('(a) FP 1', m['FP'], 1); check('(a) MOTA 0.5', m['MOTA'], 0.5, 1e-4)
    check('(a) Frag 1', m['Frag'], 1)
    # (b) C2: kf1 에서 A 가 800px(present 아님) — id2 는 제외 매칭으로 빠지고 id1(표면 전부 무시 GT·void) 도 제외
    kfs = [(10, [(1, A), (9, B)]), (20, [(1, part75), (2, A), (9, B)]), (30, [(1, part75), (2, A), (9, B)])]
    vis = dict(ep_index=np.array([0, 0, 0, 1, 1, 1]), frame=np.array([10, 20, 30] * 2),
               px_crop=np.array([5000, 800, 5000, 5000, 5000, 5000]))
    m, _ = e2e(tmp, 'cont_b', gt, kfs, vis)
    check('(b) IDSW 1', m['IDSW'], 1); check('(b) MOTA 0.6', m['MOTA'], 0.6, 1e-4)


@guarded
def test_frag_absent(tmp):
    print('E5. [감사 A-5] Frag — GT 가 present 아닌(<1600px) keyframe 을 사이에 두고 다시 추적되면 끊김 1 (TrackEval clear.py:102-107)')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); B = TS.plane(3, 0, 0.0, 50, 50)
    gt = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A), dict(oid=2, first=0, last=100, pts=B)])
    fr = [10, 20, 30, 40]
    vis = dict(ep_index=np.array([0] * 4 + [1] * 4), frame=np.array(fr * 2), px_crop=np.array([5000, 5000, 800, 5000] + [5000] * 4))
    m, _ = e2e(tmp, 'frag', gt, [(f, [(1, A), (9, B)]) for f in fr], vis)
    check('Frag 1', m['Frag'], 1); check('IDSW 0', m['IDSW'], 0); check('FP 0 (present 아닌 GT 에 매칭된 관측은 제외)', m['FP'], 0)
    check('MOTA 1', m['MOTA'], 1.0, 1e-4)


@guarded
def test_empty_timestep_keeps_continuity(tmp):
    print('E6. TrackEval clear.py:70-76 — 검출이 하나도 없는 timestep 은 연속성 기억을 지우지 않는다 (공식 구현 동작)')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); far = TS.plane(40, 40, 0.0, 20, 25)
    part75 = np.concatenate([A[:1500], far])
    gt = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A)])
    # kf0 id1(13/21) 만 → 매칭 · kf1 평가 대상 관측 없음(벽만) → 건너뜀 · kf2 id1(13/21 + 연속성)·id2(1.0) → 직전 id1 유지
    kfs = [(10, [(1, part75)]), (20, [(1, far)]), (30, [(1, part75), (2, A)])]
    m, _ = e2e(tmp, 'emptyts', gt, kfs, vis_all(1, [10, 20, 30]))
    check('IDSW 0', m['IDSW'], 0); check('Frag 0', m['Frag'], 0); check('TP 2', m['TP'], 2); check('FN 1', m['FN'], 1)
    check('FP 1', m['FP'], 1)


@guarded
def test_mt_boundary(tmp):
    print('E7. MT 경계 — tracked ratio 정확히 0.8 은 MT (MOT16 §4.1.6 "at least 80%"; TrackEval clear.py:119 는 > 라 PT)')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); far = TS.plane(40, 40, 0.0, 20, 25)
    gt = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A)])
    fr = [10, 20, 30, 40, 50]
    m, _ = e2e(tmp, 'mt', gt, [(f, [(1, A if i < 4 else far)]) for i, f in enumerate(fr)], vis_all(1, fr))
    check('MT/PT/ML = 1/0/0', (m['MT'], m['PT'], m['ML']), (1, 0, 0))
    check('경계 트랙 수 1', m.get('MT_ratio_eq_80'), 1)


@guarded
def test_tau_column(tmp):
    print('E8. [감사 A-11] tau≠0.2 로 채점해도 score_mot 이 score.json 의 열 이름(gt_iou@10)으로 읽는다')
    TS = _ts()
    A = TS.plane(0, 0, 0.0, 50, 50); far = TS.plane(40, 40, 0.0, 20, 25)
    gt = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A)])
    kfs = [(10, [(1, A + [0, 0, 0.16]), (2, np.concatenate([A[:1500], far]))])]
    # τ10: id1 은 16cm 밖 → void 100% 제외 · id2 IoU@10 = 0.75·0.68/(1.43−0.51) = 0.554 → TP
    m, out = e2e(tmp, 'tau10', gt, kfs, vis_all(1, [10]), tau=0.1)
    check('τ10: id1 제외 · id2 TP → TP 1 FP 0', (m['TP'], m['FP']), (1, 0))
    check('τ10: params.tau_m', out['params'].get('tau_m'), 0.1)
    check('τ10: params.iou_column', out['params'].get('iou_column'), 'gt_iou@10')
    m, _ = e2e(tmp, 'tau20', gt, kfs, vis_all(1, [10]), tau=0.2)
    check('τ20: id1 TP · id2 FP', (m['TP'], m['FP']), (1, 1))


@guarded
def test_old_csv_message(tmp):
    print('E10. 옛 형식 score_observations.csv (gt_iou 열 없음) → 다시 채점하라는 메시지로 멈춤')
    d = tmp / 'oldcsv'; d.mkdir()
    TS = _ts()
    TS.write_run(d, [(10, [(1, TS.plane(0, 0, 0, 5, 5))])], 100)
    (d / 'score.json').write_text('{"params": {"tau_m": 0.2, "min_frac": 0.5, "counts_column": "gt_counts@20"}}')
    (d / 'score_observations.csv').write_text('kf,frame,obs,tracklet_id,n_vox,gt_counts@20\n0,10,0,1,25,1:25\n')
    gt = TS.FakeGT([dict(oid=1, first=0, last=100, pts=TS.plane(0, 0, 0, 5, 5))])
    write_gt_h5(d / 'gt.h5', gt)
    np.savez(d / 'vis.npz', **vis_all(1, [10]))
    try:
        S.load_inputs(d, d / 'gt.h5', d / 'vis.npz')
        check('SystemExit', 'no exit', 'SystemExit')
    except SystemExit as e:
        check('SystemExit 메시지에 score_frontend', 'score_frontend' in str(e), True)


def run_e2e():
    with tempfile.TemporaryDirectory() as d:
        for fn in (test_idf1_hota_duplicates, test_similarity_is_iou, test_fragment_mot, test_fp_merge, test_fp_ignore_regions,
                   test_continuity_prev_timestep, test_frag_absent, test_empty_timestep_keeps_continuity, test_mt_boundary,
                   test_tau_column, test_old_csv_message):
            fn(Path(d))


if __name__ == '__main__':
    guarded(main)()
    guarded(test_build_data)()
    run_e2e()
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)
