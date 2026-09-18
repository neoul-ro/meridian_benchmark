#!/usr/bin/env python3
"""report.py · build_viewer.py · terms.py 자체 검증 (감사 C1 · C6 머리말 · C8 · C12 · C13 · C14 · C15 · 미검출 원인 표).

합성 입력만 쓴다 (실제 데이터·GPU 없음, 수 초).
  C1   난이도별 트랙 재현율의 모집단 = 채점 대상(eligible) GT 트랙(max_px_crop ≥ area_min_px) 만.
       등급 = score_2d_gt.csv 의 채점 대상(eligible) 행에 붙은 난이도만으로 고른 최솟값.
  C6   summary.md 머리말의 3D 매칭 설명은 score.json params 에서 만든다 (하드코딩 금지).
  C15  개수를 들고 있다가 표시할 때 한 번만 반올림: 168/204 = 82.35…% → '82.4%' (옛 코드: round(4) 후 다시 → '82.3%').
  C12  뷰어 물체 색(status)과 이유(rc)는 규칙 하나에서 나온다 — status == 'merged' ⇔ 이유가 과소분할.
  C13  뷰어 accuracy = 채점 대상 GT 트랙의 탐지 인정(detect_credit=1) 매칭 관측의 관측별 평균 거리(dist_mean_cm)의 중앙값
       (summary.md 표의 per_obs_mean 과 같은 통계).
  C14  매칭 관측 수도 채점 대상 GT 트랙 · 탐지 인정 관측만. 'keyframe 중 최대 가시 px' 와 '전 프레임 최대 px' 를 구분.
  C8   keyframe 그림은 입력 내용 키(render key)가 바뀌었거나 그림이 없으면 다시 그린다.
  원인 표  미검출 원인 값 → 표준 용어는 terms.py 한 곳. 모르는 값은 눈에 띄는 대체 문구로 (조용히 사라지지 않게).
  [수정 ⑤ 연결]
  P1   report.py STEPS: 순서 labels → score3d → mot → difficulty → seg2d → geometry · score3d 산출물에 score_kf_gt.csv ·
       score3d 에 labels_dir · workers 전달 · expect_score3d 가 새 params(similarity · iou_threshold · loc_min_iou ·
       labels_version · labels_meta_sha1) 를 기대 · 라벨 meta.json 내용이 바뀌면 score3d 캐시 키가 바뀐다 (PNG 가 같아도)
  P2   원인 값 = no_kf · merged · split · low_iou · missed (absorbed · not_detected 는 표에서 뺌 → 대체 문구) ·
       (keyframe, GT) 상태 tp · merged · split · low_iou · missed · ignored_match · not_present 표시 이름
  P3   summary.md 머리말에 IoU_τ 정의(params) · '검토 중/바뀌는 중' 없음 · 예측 검출 상태·TIDE 오류 표 ·
       README · 뷰어 · status_md 에 옛 share 매칭 설명·임시 문구 없음
  P4   뷰어 물체 정보: (keyframe, GT) 3D 상태 개수 (build_viewer.kf3d_counts)
"""
import csv
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


def write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def level_fixture(d, n_eps=None):
    """C1 합성: 에피소드 4개.
      ep0 max_px 800  (채점 대상 아님)  탐지 · 등급 행: (f10, easy, eligible)
      ep1 max_px 2000 (대상)            탐지 · 등급 행: (f10, easy, eligible=False) · (f20, moderate, eligible)
      ep2 max_px 5000 (대상)            놓침 · 등급 행: (f20, hard, eligible)
      ep3 max_px 3000 (대상)            놓침 · 등급 행: (f30, easy, eligible=False) 뿐
    기대: easy n=0 · moderate {ep1} 1/1 · hard {ep1, ep2} 1/2
    옛 코드(모든 등급 행 · 모든 에피소드): easy {ep0, ep1, ep3} 2/3"""
    run = Path(d); run.mkdir(parents=True, exist_ok=True)
    write_csv(run / 'score_episodes.csv', [
        dict(ep_index=0, detected=1, max_px_crop=800, miss_reason=''),
        dict(ep_index=1, detected=1, max_px_crop=2000, miss_reason=''),
        dict(ep_index=2, detected=0, max_px_crop=5000, miss_reason='missed'),
        dict(ep_index=3, detected=0, max_px_crop=3000, miss_reason='missed')])
    write_csv(run / 'score_2d_gt.csv', [
        dict(ep_index=0, frame=10, kf=0, eligible=True, status='tp', level=0),
        dict(ep_index=1, frame=10, kf=0, eligible=False, status='missed', level=0),
        dict(ep_index=1, frame=20, kf=1, eligible=True, status='tp', level=1),
        dict(ep_index=2, frame=20, kf=1, eligible=True, status='missed', level=2),
        dict(ep_index=3, frame=30, kf=2, eligible=False, status='missed', level=0)])
    np.savez(run / 'difficulty.npz', frame=np.array([10, 10, 20, 20, 30]), ep_index=np.array([0, 1, 1, 2, 3]),
             level=np.array([0, 0, 1, 2, 0], np.int8))
    (run / 'score.json').write_text(json.dumps(dict(visibility=dict(area_min_px=1600))))
    return run


def fake_block(level_3d=None, params=None, reasons=None):
    """report.render 가 읽는 키를 모두 가진 합성 시퀀스 블록 (옛 render 가 읽는 키 포함)."""
    p = dict(tau_m=0.2, min_frac=0.5, absorb_frac=0.2, voxel_m=0.02, max_range_m=5.0,
             matching='hungarian 1:1 per keyframe (cost = voxel share within tau, no temporal term)', dup_policy='hungarian',
             window_margin_frames=5, sensitivity_taus_m=[0.1, 0.2, 0.5], dist_eps_m=1e-5, counts_column='gt_counts@20',
             detect_window='keyframes where the GT track is present (label px within 5m >= 1600)',
             label_window='not used for matching since 26/09/17',
             similarity='tolerant IoU_tau = P*R/(P+R-P*R)', iou_threshold=0.5, loc_min_iou=0.25, labels_version='v-test',
             presence='label px within 5m >= 1600 at that keyframe', fp_rule='FP-RULE-TEXT', dup_rule='DUP-RULE-TEXT',
             miss_reasons='MISS-REASONS-TEXT', merge_split_coverage='COVERAGE-TEXT')
    p.update(params or {})
    rs = reasons if reasons is not None else {'merged': 16, 'missed': 22}
    kind = lambda n, r: dict(n_detectable=n, recall_detectable=r, miss_reasons_detectable=rs if n else {})
    b = dict(
        params=p, sequence='fake', frames=[0, 100],
        frontend=dict(n_kf=204, n_obs=4600, n_tracklet_ids=1320, points_dropped_beyond_range=0.0712, frames=1779,
                      ms_per_frame=8.13, overflow_frames=0),
        missed=dict(n_gt_episodes=483, detected=248, recall=0.5135, missed=235, miss_reasons={}),
        visibility=dict(area_min_px=1600, n_in_view=477, recall_in_view=0.5199, n_detectable=245, recall_detectable=0.8449,
                        by_kind=dict(static=kind(245, 0.8449), human=kind(0, None)),
                        recall_by_max_px_crop=[dict(bin='0 (시야 밖)', n=6, recall=0.0), dict(bin='1600-6399', n=126, recall=0.746)],
                        keyframe_level=dict(n_pairs=2247, recall=0.7143, by_kind=dict(static=dict(n=2247, recall=0.7143),
                                                                                     human=dict(n=0, recall=None)))),
        accuracy=dict(n_obs_matched=1887, obs_match_rate=0.4102, n_obs_duplicate=1387, duplicate_rate=0.3015,
                      dist_error_cm=dict(per_obs_mean=dict(p50=0.88, p90=13.5), per_obs_median=dict(p25=0, p50=0, p75=0, p90=5)),
                      within_tau_fraction_mean=0.945, voxel_exact_2cm_mean=0.801, multi_object_obs_rate=0.134,
                      episode_coverage_at_tau=dict(p25=0.5, p50=0.8, p75=0.95), n_obs_tp=1700, n_obs_ignored_match=187,
                      n_obs_fp=1500, n_obs_excluded=1213, unmatched_tide_errors=dict(dupe=900, loc=700, bkg=1113)),
        keyframe_gt=dict(n_present=2247, status=dict(tp=1400, merged=300, split=80, low_iou=200, missed=267, ignored_match=187,
                                                     not_present=5000)),
        sensitivity=[dict(tau_cm=10, recall=0.49, obs_match_rate=0.39), dict(tau_cm=20, recall=0.51, obs_match_rate=0.41),
                     dict(tau_cm=50, recall=0.52, obs_match_rate=0.42)])
    if level_3d is not None:
        b['level_3d'] = level_3d
    return b


def section(md, start):
    """start 줄부터 다음 '## ' 제목 전까지."""
    lines = md.splitlines()
    i = next((k for k, l in enumerate(lines) if l.startswith(start)), None)
    if i is None:
        return ''
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith('## ')), len(lines))
    return '\n'.join(lines[i:j])


# ------------------------------------------------------------------ C1
def test_c1_level_population(tmp):
    """C1 난이도별 트랙 재현율 모집단 = 채점 대상 GT 트랙, 등급 = 채점 대상(eligible) 2D 행만"""
    import report as R
    run = level_fixture(tmp / 'c1')
    lv = R.recall_3d_by_level(run, run / 'difficulty.npz')
    check('easy n', lv['easy']['n'], 0)
    check('moderate n', lv['moderate']['n'], 1)
    check('moderate 탐지 수', lv['moderate'].get('n_detected'), 1)
    check('hard n', lv['hard']['n'], 2)
    check('hard 탐지 수', lv['hard'].get('n_detected'), 1)


# ------------------------------------------------------------------ C15
def test_c15_round_once(tmp):
    """C15 168/204 → 82.4% (한 번만 반올림)"""
    import report as R
    run = Path(tmp / 'c15'); run.mkdir(parents=True, exist_ok=True)
    eps = [dict(ep_index=i, detected=int(i < 168), max_px_crop=2000, miss_reason='' if i < 168 else 'missed') for i in range(204)]
    write_csv(run / 'score_episodes.csv', eps)
    write_csv(run / 'score_2d_gt.csv', [dict(ep_index=i, frame=i, kf=i, eligible=True, status='tp', level=0) for i in range(204)])
    np.savez(run / 'difficulty.npz', frame=np.arange(204), ep_index=np.arange(204), level=np.zeros(204, np.int8))
    (run / 'score.json').write_text(json.dumps(dict(visibility=dict(area_min_px=1600))))
    lv = R.recall_3d_by_level(run, run / 'difficulty.npz')
    md = R.render({'apartment_s1_00h': fake_block(level_3d=lv)})
    rows = [l for l in md.splitlines() if l.startswith('| apartment_00h |') and ('(n=204' in l or '/204' in l)]
    easy_cells = [l.split('|')[2] for l in rows]
    check('Easy 칸에 82.4% (168/204 = 82.3529…%)', any('82.4%' in c for c in easy_cells), True)
    check('Easy 칸에 82.3% 없음', any('82.3%' in c for c in easy_cells), False)


# ------------------------------------------------------------------ C6 머리말
def test_c6_header_from_params(tmp):
    """C6 3D 매칭 머리말 = score.json params (τ · dup_policy · matching 문자열)"""
    import report as R
    md = R.render({'seqA': fake_block(params=dict(matching='MATCH-DESC-XYZ', tau_m=0.1, dup_policy='count'))})
    head = section(md, '## 8.')          # M6: 정의 · params 는 맨 뒤 부록으로 옮겼다
    check('params 절에 params.matching 문자열', 'MATCH-DESC-XYZ' in head, True)
    check('params 절에 τ = 10cm', '10cm' in head.replace(' cm', 'cm'), True)
    check('params 절에 dup_policy count', 'count' in head, True)
    check('params 절에 하드코딩된 "직전 id 우선" 없음', '직전 id 우선' in head, False)
    check('[P3] params 절에 IoU_τ 임계 0.5 · Loc 0.25 · 라벨 버전', all(x in head for x in ('0.5', '0.25', 'v-test')), True)
    check('[P3] params 절에 FP · 중복 · 원인 · 덮음 규칙 문자열', all(x in head for x in ('FP-RULE-TEXT', 'DUP-RULE-TEXT', 'MISS-REASONS-TEXT', 'COVERAGE-TEXT')), True)
    check('[P3] params 절에 "바뀌는 중" · "검토 중" 없음', ('바뀌는 중' in head) or ('검토 중' in head), False)


# ------------------------------------------------------------------ 미검출 원인 표
def test_reason_table(tmp):
    """[P2] 미검출 원인 값 → 표준 용어 (terms.py 한 곳) · 옛 값 absorbed·not_detected 는 표에 없음 · 모르는 값 대체 문구"""
    import terms as T
    check('merged → 과소분할', '과소분할' in T.reason_label('merged'), True)
    check('split → 과다분할', '과다분할' in T.reason_label('split'), True)
    check('low_iou → Loc', 'Loc' in T.reason_label('low_iou'), True)
    check('no_kf → keyframe 없음', 'keyframe 없음' in T.reason_label('no_kf'), True)
    check('missed → Miss', 'Miss' in T.reason_label('missed'), True)
    check('원인 표 = no_kf · merged · split · low_iou · missed', sorted(T.REASONS), sorted(['no_kf', 'merged', 'split', 'low_iou', 'missed']))
    check('absorbed (옛 값) → 대체 문구', '알 수 없는' in T.reason_label('absorbed'), True)
    check('not_detected (옛 값) → 대체 문구', '알 수 없는' in T.reason_label('not_detected'), True)
    want = ['tp', 'merged', 'split', 'low_iou', 'missed', 'ignored_match', 'not_present']
    check('(keyframe, GT) 3D 상태 표 = 7개', list(T.KF3D), want)
    check('3D 상태 표시 이름에 대체 문구 없음', [k for k in want if '알 수 없는' in T.kf3d_label(k)], [])
    check('ignored_match → 무시 GT 매칭 (M5: 제외는 채점 제외 한 뜻)', '무시 GT 매칭' in T.kf3d_label('ignored_match'), True)
    lab = T.reason_label('zzz_new_reason')
    check('모르는 값: 원래 값이 보임', 'zzz_new_reason' in lab, True)
    check('모르는 값: 대체 문구 표시', '알 수 없는' in lab, True)
    import report as R
    md = R.render({'seqA': fake_block(reasons={'zzz_new_reason': 3, 'merged': 1})})
    check('summary.md 원인 칸에 모르는 값이 대체 문구로', 'zzz_new_reason' in md and '알 수 없는' in md, True)
    check('summary.md 원인 칸에 코드값 merged 가 그대로 노출 안 됨', '| merged' in md or ' merged 1' in md, False)
    check('[P3] summary.md 에 (keyframe, GT) 상태 표 (무시 GT 매칭 187 · present 아님 5000)', '| 187 | 5000 |' in md, True)
    check('[P3] summary.md 에 TIDE 오류 표 (Dupe 900 · Loc 700 · Bkg 1113)', '| 900 | 700 | 1113 |' in md, True)


# ------------------------------------------------------------------ C12
def ep(e, px, det, reason):
    return dict(ep_index=str(e), max_px_crop=str(px), detected=str(det), miss_reason=reason, first_frame='0', last_frame='9')


def test_c12_single_rule(tmp):
    """C12 색(status)과 이유(rc)를 한 규칙으로 — apartment_01h 물체 13 사례 + 모든 조합"""
    import build_viewer as BV
    import itertools
    v = BV.object_verdict([ep(0, 57404, 0, 'missed'), ep(1, 23900, 0, 'merged')])
    check('물체 13: status merged', v['status'], 'merged')
    check('물체 13: 이유 코드 merged', v['rc'], 'merged')
    bad = 0
    codes = ('no_kf', 'merged', 'split', 'low_iou', 'missed', 'zzz')
    for n in (1, 2, 3):
        for combo in itertools.product(codes, repeat=n):
            vv = BV.object_verdict([ep(i, 5000, 0, c) for i, c in enumerate(combo)])
            under = vv['rc'] == 'merged'
            if (vv['status'] == 'merged') != under or vv['status'] not in ('merged', 'missed') or vv['rc'] not in combo:
                bad += 1
    check('모든 조합에서 status==merged ⇔ 이유가 과소분할, 이유는 실제 에피소드 값 중 하나', bad, 0)
    check('채점 대상 에피소드가 없으면 ineligible', BV.object_verdict([ep(0, 800, 1, '')])['status'], 'ineligible')
    check('대상 중 하나라도 탐지면 detected', BV.object_verdict([ep(0, 800, 0, 'no_kf'), ep(1, 1600, 1, '')])['status'], 'detected')


# ------------------------------------------------------------------ C13 · C14
def test_c13_c14_object_metrics(tmp):
    """C13 accuracy 통계 · C14 매칭 관측 수(대상 · 인정만) · keyframe 최대 px 와 전 프레임 최대 px 구분"""
    import build_viewer as BV
    erows = [ep(0, 5000, 1, ''), ep(1, 900, 1, '')]                    # ep1 은 채점 대상 아님
    obs_by_ep = {0: [dict(dist_mean_cm='0.5', dist_med_cm='0.0', detect_credit='1'),
                     dict(dist_mean_cm='0.7', dist_med_cm='0.0', detect_credit='1'),
                     dict(dist_mean_cm='9.0', dist_med_cm='9.0', detect_credit='0')],      # 판정 창 밖
                 1: [dict(dist_mean_cm='5.0', dist_med_cm='5.0', detect_credit='1')]}
    best_kf_ep = {0: (1200, 3), 1: (900, 7)}
    m = BV.object_metrics([0, 1], erows, obs_by_ep, best_kf_ep)
    check('C14 매칭 관측 수 = 대상 에피소드의 인정 관측만 (2)', m['n_matched'], 2)
    check('C13 accuracy = 관측별 평균 거리의 중앙값 (0.6)', m['dist_cm'], 0.6)
    check('C14 keyframe 중 최대 가시 px (대상 에피소드)', m['best_kf_px'], 1200)
    check('C14 전 프레임 최대 가시 px (대상 에피소드)', m['max_px_all'], 5000)
    check('C14 best keyframe 번호', m['best_kf'], 3)
    html = (HERE / 'viewer_template.html').read_text()
    check('뷰어: 옛 "최대 가시" 라벨 없음', '<dt>최대 가시</dt>' in html, False)
    check('뷰어: 옛 "cm (중앙값)" 라벨 없음', "cm (중앙값)" in html, False)


# ------------------------------------------------------------------ C8
def test_c8_render_staleness(tmp):
    """C8 keyframe 그림 다시 그리기 판정 — 입력 키가 바뀌었거나 그림/기록이 없으면 stale"""
    import build_viewer as BV
    kf = tmp / 'c8' / 'kf'; kf.mkdir(parents=True)
    (kf / '000000.jpg').write_bytes(b'x'); (kf / '000034.jpg').write_bytes(b'x')
    BV.save_render_manifest(kf, {0: 'key-a', 34: 'key-c'})
    check('키 같음 · 그림 있음 → 안 그림', BV.stale_kf_frames(kf, {0: 'key-a', 34: 'key-c'}), [])
    check('frame 0 키 바뀜(2D 상태 변경) → 0 만', BV.stale_kf_frames(kf, {0: 'key-b', 34: 'key-c'}), [0])
    (kf / '000034.jpg').unlink()
    check('그림 없어짐 → 그 프레임', BV.stale_kf_frames(kf, {0: 'key-a', 34: 'key-c'}), [34])
    check('기록에 없는 프레임 → 그림 있어도 다시', BV.stale_kf_frames(kf, {0: 'key-a', 46: 'key-d'}), [46])
    gt1 = [dict(ep_index='7', status='tp', eligible='True')]; gt2 = [dict(ep_index='7', status='missed', eligible='True')]
    pr = [dict(pred='0', status='tp')]
    k1 = BV.kf_render_key(0, gt1, pr, 'labsha', 'h5sha', 'code'); k2 = BV.kf_render_key(0, gt2, pr, 'labsha', 'h5sha', 'code')
    check('2D GT 상태 tp→missed 면 키가 바뀜 (감사 a14 사례)', k1 != k2, True)
    check('라벨 PNG 내용이 바뀌면 키가 바뀜', k1 != BV.kf_render_key(0, gt1, pr, 'labsha2', 'h5sha', 'code'), True)


# ------------------------------------------------------------------ P1 단계 그래프 연결
class _Ctx:
    def __init__(self, d):
        import provenance as PV
        d = Path(d)
        self.s = 'apartment_s1_00h'; self.name = f'uHumans2_{self.s}'; self.workers = 7
        self.run = d / 'run'; self.labels = d / 'labels'; self.seqd = d / 'seq'
        for p in (self.run, self.labels, self.seqd):
            p.mkdir(parents=True, exist_ok=True)
        self.gt = d / 'gt.h5'; self.vis = d / 'vis.npz'; self.h5 = self.run / 'frontend_output.h5'
        for p in (self.gt, self.vis, self.h5):
            p.write_bytes(b'x')
        self.kf_frames = np.array([0, 5]); self.has_masks = True
        self.kf_digest = PV.sha1_text('0,5')
        (self.labels / 'meta.json').write_text(json.dumps(dict(version='v1')))

    def label_pngs(self):
        return [self.labels / '000000.png', self.labels / '000005.png']


def test_p1_steps(tmp):
    """P1 report.py 단계 그래프 — 순서 · score_kf_gt.csv · labels_dir/workers · 새 params · 라벨 출처가 캐시 키에"""
    import report as R
    import provenance as PV
    c = _Ctx(tmp / 'p1')
    steps = R.STEPS(c)
    check('실행 순서', [st['name'] for st in R.topo(steps)], ['labels', 'score3d', 'mot', 'difficulty', 'seg2d', 'geometry'])
    s3 = next(st for st in steps if st['name'] == 'score3d')
    check('score3d 산출물에 score_kf_gt.csv', [p.name for p in s3['outputs']()],
          ['score.json', 'score_episodes.csv', 'score_observations.csv', 'score_kf_gt.csv'])
    check('mot 는 score3d 뒤 (deps)', next(st for st in steps if st['name'] == 'mot')['deps'], ('score3d',))
    got = {}
    orig = R.S.score
    try:
        R.S.score = lambda *a, **kw: got.update(kw)
        s3['run'](c)
    finally:
        R.S.score = orig
    check('score3d 에 labels_dir · workers 전달', (got.get('labels_dir'), got.get('workers')), (c.labels, 7))
    e = R.expect_score3d(c)
    for k in ('similarity', 'iou_threshold', 'loc_min_iou', 'labels_version', 'labels_meta_sha1', 'tau_m', 'dup_policy'):
        check(f'expect_score3d 에 {k}', k in e, True)
    import hashlib
    check('labels_meta_sha1 = meta.json sha1', e.get('labels_meta_sha1'), hashlib.sha1((c.labels / 'meta.json').read_bytes()).hexdigest())
    check('labels_version = meta.json version', e.get('labels_version'), 'v1')
    H = PV.Hasher(tmp / 'p1' / 'sha1.json')
    run = R.Runner(c, H, force=False)
    k1 = run.parts(s3)
    (c.labels / 'meta.json').write_text(json.dumps(dict(version='v1', report_provenance=dict(finished='other'))))
    k2 = run.parts(s3)
    check('라벨 meta.json 내용이 바뀌면 score3d 키가 바뀐다 (PNG 같아도)', k1 != k2, True)
    j = dict(params=dict(R.expect_score3d(c)), visibility=dict(area_min_px=1600))
    (c.run / 'score.json').write_text(json.dumps(j))
    check('check_score3d: 기대와 같으면 불일치 없음', R.check_score3d(c), [])
    j['params']['labels_meta_sha1'] = 'old'
    (c.run / 'score.json').write_text(json.dumps(j))
    check('check_score3d: 옛 라벨로 채점한 score.json → 불일치', any('labels_meta_sha1' in b for b in R.check_score3d(c)), True)


def test_p3_texts(tmp):
    """P3 README · 뷰어 · status_md · report 에 임시 문구·옛 share 매칭 설명 없음"""
    import re
    files = {n: (HERE / n).read_text() for n in ('README.md', 'viewer_template.html', 'status_md.py', 'report.py', 'terms.py', 'build_viewer.py')}
    for n in ('docs/METRICS.md', 'docs/HISTORY.md'):
        if (HERE / n).exists():
            files[n] = (HERE / n).read_text()
    bad = {}
    for n, t in files.items():
        hits = [w for w in ('검토 중', '바뀌는 중', '바꾸는 중', '구현 중', 'share 합 최대', r'\babsorbed\b', r'\bnot_detected\b', 'min_frac 이상 겹친')
                if re.search(w, t)]
        if hits:
            bad[n] = hits
    check('임시 문구 · 옛 원인 값 · share 매칭 설명 없음', bad, {})
    met = (HERE / 'docs' / 'METRICS.md').read_text() if (HERE / 'docs' / 'METRICS.md').exists() else ''
    for w in ('IoU_τ', 'P·R', '1600', 'panopticapi', 'TIDE', 'quantify.py', 'nearest', 'ignored_match', 'not_present'):
        check(f'docs/METRICS.md 3D 매칭 절에 {w}', w in met, True)


def test_p4_viewer_kf3d(tmp):
    """P4 뷰어 물체 정보의 (keyframe, GT) 3D 상태 개수"""
    import build_viewer as BV
    rows = [dict(ep_index='1', status='tp'), dict(ep_index='1', status='merged'), dict(ep_index='2', status='not_present'),
            dict(ep_index='1', status='merged'), dict(ep_index='3', status='ignored_match')]
    c = BV.kf3d_counts(rows)
    check('에피소드별 상태 개수', (c[1], c[2], c[3]), ({'tp': 1, 'merged': 2}, {'not_present': 1}, {'ignored_match': 1}))
    check('물체 합산 (ep 1, 2)', BV.object_kf3d([1, 2], c), {'tp': 1, 'merged': 2, 'not_present': 1})
    html = (HERE / 'viewer_template.html').read_text()
    check('뷰어 템플릿이 kf3d 를 보여 줌', 'kf3d' in html, True)


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for fn in (test_c1_level_population, test_c15_round_once, test_c6_header_from_params, test_reason_table,
                   test_c12_single_rule, test_c13_c14_object_metrics, test_c8_render_staleness, test_p1_steps, test_p3_texts,
                   test_p4_viewer_kf3d):
            print(f'[{fn.__name__}] {fn.__doc__.strip().splitlines()[0]}')
            try:
                fn(tmp)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(limit=3)
                print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
                FAILS.append(fn.__name__)
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
