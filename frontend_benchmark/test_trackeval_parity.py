#!/usr/bin/env python3
"""score_mot.py 지표 ↔ TrackEval 공식 구현(CLEAR · Identity · HOTA) 대조.
(py-motmetrics 의 motmetrics/tests/test_trackeval_parity.py 와 같은 취지.)

같은 입력 dict(timestep 별 GT id · 예측 id · similarity)를 두 구현에 넣고 지표를 비교한다.
  TrackEval 경로  trackeval_path.resolve() — 환경변수 TRACKEVAL_PATH (local.env 도 됨) → pip 로 설치된 trackeval →
                  <WS>/third_party/TrackEval. 기본값에 내부 감사 폴더를 두지 않는다 (사용성 리뷰 M7). 고정 commit 12c8791.
                  없거나 import 가 안 되면 받는 법(trackeval_path.HOWTO)을 적고 skip (종료코드 0).
                  numpy ≥ 1.24 에서 사라진 np.float·np.int·np.bool 별칭은 여기서만 채운다.
  비교 지표       MOTA · IDSW · Frag · IDF1 · IDTP · IDFP · IDFN · DetA · AssA · HOTA (단일 α) + TP · FN · FP · HOTA_TP.
                  실수는 |차이| ≤ 1e-6, 정수는 같아야 한다.
  예외 (문서화된 정의 차이만)
    · MT/PT: 우리는 MOT16 §4.1.6 "at least 80%"(≥), TrackEval clear.py:119 는 > 0.8.
             → 우리 MT − TrackEval MT == MT_ratio_eq_80, PT 는 그만큼 반대, ML 은 같아야 한다.
    · HOTA TP 가 0 이면 우리 AssA = null, TrackEval = 0.
사례
  1. 끝에서 끝까지 합성 (score_frontend → score_mot.build_data): 감사 C1·C2·C3·C4·C11 · 검출 없는 timestep
     · 무시 GT(present 아님 · 사람) 매칭 · 작은 순수 조각 (26/09/17 결정 ④: similarity = IoU_τ)
  2. 무작위 입력 60개: 빈 timestep · 중복 검출 · 정확히 α 인 유사도 · id 재등장 · 조각끼리 번갈아 겹침
  3. 실제 apartment_s1_00h (에피소드 · 물체 단위). 새 형식(gt_iou@20 열) score_observations.csv 를 찾는 순서:
     PARITY_SCORE_DIR → <REAL_RUNS>/uHumans2_apartment_s1_00h → <REAL_RUNS>/_audit/fix_D_match3d/apartment_s1_00h
     → 없으면 임시 폴더에 score_frontend 로 다시 채점 (최신 버전 GT 라벨이 있을 때만, 없으면 SKIP). REAL_RUNS = paths.REAL_RUNS
  4. 전처리 대조: 무시 GT 에 매칭된 예측 제거를 TrackEval MotChallenge2DBox.get_preprocessed_seq_data
     (mot_challenge_2d_box.py:370-387, 무시 GT = distractor 클래스 8)와 비교 — 1·3 의 모든 사례에서 timestep 마다 남는 GT·예측이 같은지.
     TrackEval MOT17 전처리에는 '매칭 안 된 예측의 무시 영역 비율' 단계가 없어서(그 단계는 kitti_mots.py:336-344),
     그 단계로 우리가 뺀 예측(mostly_ignore_region)은 두 쪽 입력에서 똑같이 미리 뺀다.
실행: python test_trackeval_parity.py [--no-real]
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_mot as SM  # noqa: E402
import trackeval_path as TEP  # noqa: E402
from paths import WS, REAL_RUNS, gt_h5, seq_dir  # noqa: E402

TOL = 1e-6
FAILS = []


def run_dir(s):
    return REAL_RUNS / f'uHumans2_{s}'          # 실제 데이터 — FB_RUNS(출력 폴더)와 무관


def vis_npz(s):
    return REAL_RUNS / 'gt_vis' / f'uHumans2_{s}.npz'


def import_trackeval():
    where, how = TEP.resolve()
    if where is None:
        print(f'  SKIP TrackEval 사본이 없습니다 ({how}). {TEP.HOWTO}')
        return None
    for name, typ in (('float', float), ('int', int), ('bool', bool)):   # TrackEval 12c8791 은 옛 numpy 별칭을 쓴다
        if name not in np.__dict__:
            setattr(np, name, typ)
    sys.path.insert(0, str(where))
    try:
        with redirect_stdout(io.StringIO()):                           # 'Error importing BURST ...' 안내문 숨김
            from trackeval.metrics import CLEAR, HOTA, Identity
    except Exception as e:  # noqa: BLE001
        print(f'  SKIP TrackEval import 실패 ({where}, {how}): {type(e).__name__}: {e}. {TEP.HOWTO}')
        return None
    return CLEAR, Identity, HOTA


def run_trackeval(TE, data, alpha):
    CLEAR, Identity, HOTA = TE
    clear = CLEAR({'THRESHOLD': alpha, 'PRINT_CONFIG': False})
    ident = Identity({'THRESHOLD': alpha, 'PRINT_CONFIG': False})
    hota = HOTA()
    hota.array_labels = np.array([alpha])
    c, i, h = clear.eval_sequence(data), ident.eval_sequence(data), hota.eval_sequence(data)
    return dict(TP=int(c['CLR_TP']), FN=int(c['CLR_FN']), FP=int(c['CLR_FP']), IDSW=int(c['IDSW']), Frag=int(c['Frag']),
                MT=int(c['MT']), PT=int(c['PT']), ML=int(c['ML']), MOTA=float(c['MOTA']),
                IDTP=int(i['IDTP']), IDFP=int(i['IDFP']), IDFN=int(i['IDFN']), IDF1=float(i['IDF1']),
                DetA=float(h['DetA'][0]), AssA=float(h['AssA'][0]), HOTA=float(h['HOTA'][0]), HOTA_TP=int(h['HOTA_TP'][0]))


INTS = ('TP', 'FN', 'FP', 'IDSW', 'Frag', 'IDTP', 'IDFP', 'IDFN', 'HOTA_TP')
FLOATS = ('MOTA', 'IDF1', 'DetA', 'AssA', 'HOTA')


def compare(name, TE, data, alpha=0.5, show=False):
    """반환: 표 행 목록 [(지표, 우리, TrackEval, 차이)]. 실패는 FAILS 에."""
    ours, _ = SM.mot_metrics(data, alpha)
    te = run_trackeval(TE, data, alpha)
    bad, table = [], []
    for k in INTS:
        table.append((k, ours[k], te[k], ours[k] - te[k]))
        if ours[k] != te[k]:
            bad.append(k)
    for k in FLOATS:
        o, t = ours[k], te[k]
        if k == 'AssA' and ours['HOTA_TP'] == 0:          # 문서화된 차이: TP 0 → 우리 null · TrackEval 0
            table.append((k, o, t, None))
            if not (o is None and t == 0):
                bad.append(k)
            continue
        if k == 'MOTA' and data['num_gt_dets'] == 0:      # GT 0 → 우리 null (TrackEval 은 early return 0)
            table.append((k, o, t, None))
            continue
        d = None if o is None else o - t
        table.append((k, o, t, d))
        if d is None or abs(d) > TOL:
            bad.append(k)
    eq80 = ours['MT_ratio_eq_80']                         # 문서화된 차이: MT 경계 (MOT16 ≥ vs TrackEval >)
    table.append(('MT', ours['MT'], te['MT'], ours['MT'] - te['MT']))
    table.append(('PT', ours['PT'], te['PT'], ours['PT'] - te['PT']))
    table.append(('ML', ours['ML'], te['ML'], ours['ML'] - te['ML']))
    if (ours['MT'] - te['MT'], ours['PT'] - te['PT'], ours['ML'] - te['ML']) != (eq80, -eq80, 0):
        bad.append(f'MT/PT/ML (eq80={eq80})')
    ok = not bad
    print(f'  {"OK " if ok else "FAIL"} {name}' + ('' if ok else f'  다른 지표: {bad}'))
    if not ok:
        FAILS.append(f'{name}: {bad}')
    if show or not ok:
        print(f'      {"지표":8s} {"우리":>12s} {"TrackEval":>12s} {"차이":>12s}')
        for k, o, t, d in table:
            fmt = lambda x: '-' if x is None else (f'{x:.6f}' if isinstance(x, float) else str(x))  # noqa: E731
            print(f'      {k:8s} {fmt(o):>12s} {fmt(t):>12s} {fmt(d):>12s}')
    return table


def mot17_dataset():
    """TrackEval MotChallenge2DBox 의 전처리 메서드만 쓰려고 생성자 없이 만든 객체 (생성자는 파일 경로 설정을 요구한다).
    속성 값은 mot_challenge_2d_box.py:61·75-78 과 같다."""
    try:
        with redirect_stdout(io.StringIO()):
            from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    except Exception as e:  # noqa: BLE001
        print(f'  SKIP 전처리 대조: MotChallenge2DBox import 실패 {type(e).__name__}: {e}')
        return None
    ds = object.__new__(MotChallenge2DBox)
    ds.do_preproc, ds.benchmark = True, 'MOT17'
    ds.class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
                                 'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
                                 'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
    ds.valid_class_numbers = list(ds.class_name_to_class_id.values())
    return ds


_DS = []


def preproc_parity(name, TE, inp, unit, area_min_px=1600):
    """우리 build_data 결과와 TrackEval MOT17 전처리 결과가 timestep 마다 같은 GT·예측을 남기는지.
    TrackEval 입력: 후보 GT 전부(유효 = pedestrian 1, 무시 = distractor 8) · 복셀 있는 예측 중 우리 3단계(mostly_ignore_region)로
    빠지지 않은 것 · similarity = gt_iou. det 배열 0열에 원래 키를 실어 np.delete 뒤에도 무엇이 남았는지 읽는다."""
    import match3d as M
    if not _DS:
        _DS.append(mot17_dataset())
    ds = _DS[0]
    if ds is None:
        return
    ours = SM.sequence_data(inp, unit, area_min_px)
    step3 = {(k, t) for k, t, r in ours['removed'] if r == 'mostly_ignore_region'}
    gts_at, obs_at = {}, {}
    for r in inp['kf_gt']:
        gts_at.setdefault(int(r['kf']), []).append((int(r['ep_index']), int(r['px_label_5m'])))
    for r in inp['rows']:
        obs_at.setdefault(int(r['kf']), []).append(r)
    raw = dict(gt_ids=[], gt_dets=[], gt_classes=[], gt_extras=[], tracker_ids=[], tracker_dets=[], tracker_classes=[],
               tracker_confidences=[], similarity_scores=[], num_timesteps=inp['n_kf'], seq='parity')
    for k in range(inp['n_kf']):
        cand = sorted(gts_at.get(k, []))
        key = (lambda e: e) if unit == 'episode' else (lambda e: int(inp['ep_oid'][e]))
        valid = [px >= area_min_px and int(inp['ep_oid'][e]) not in inp['human_obj'] for e, px in cand]
        obs = [r for r in obs_at.get(k, []) if int(r['n_vox']) > 0 and (k, int(r['tracklet_id'])) not in step3]
        raw['gt_ids'].append(np.array([key(e) for e, _ in cand], int))
        raw['gt_dets'].append(np.array([[key(e), 0, 1, 1] for e, _ in cand], float).reshape(-1, 4))
        raw['gt_classes'].append(np.array([1 if v else 8 for v in valid], int))
        raw['gt_extras'].append({'zero_marked': np.ones(len(cand), int)})
        raw['tracker_ids'].append(np.array([int(r['tracklet_id']) for r in obs], int))
        raw['tracker_dets'].append(np.array([[int(r['tracklet_id']), 0, 1, 1] for r in obs], float).reshape(-1, 4))
        raw['tracker_classes'].append(np.ones(len(obs), int))
        raw['tracker_confidences'].append(np.ones(len(obs)))
        iou = [M.parse_values(r[inp['iou_col']]) for r in obs]
        raw['similarity_scores'].append(np.array([[d.get(e, 0.0) for d in iou] for e, _ in cand], float).reshape(len(cand), len(obs)))
    with redirect_stdout(io.StringIO()):
        te = ds.get_preprocessed_seq_data(raw, 'pedestrian')
    bad = []
    for k in range(inp['n_kf']):
        o_g = sorted(ours['gt_keys'][i] for i in ours['gt_ids'][k]); t_g = sorted(int(x) for x in np.asarray(te['gt_dets'][k])[:, 0])
        o_t = sorted(ours['tracker_keys'][i] for i in ours['tracker_ids'][k])
        t_t = sorted(int(x) for x in np.asarray(te['tracker_dets'][k]).reshape(-1, 4)[:, 0])
        if o_g != t_g or o_t != t_t:
            bad.append((k, o_g, t_g, o_t, t_t))
    ok = not bad and ours['num_tracker_dets'] == te['num_tracker_dets'] and ours['num_gt_dets'] == te['num_gt_dets']
    print(f'  {"OK " if ok else "FAIL"} 전처리 {name}: 예측 검출 {ours["num_tracker_dets"]} vs TrackEval {te["num_tracker_dets"]}'
          f' · GT 검출 {ours["num_gt_dets"]} vs {te["num_gt_dets"]} · 다른 timestep {len(bad)}' + (f' 첫 차이 {bad[0]}' if bad else ''))
    if not ok:
        FAILS.append(f'전처리 {name}')


# ---------------------------------------------------------------------------------------------------------------
def synthetic_e2e(TE, tmp):
    print('1. 끝에서 끝까지 합성 (score_frontend → score_mot.build_data)')
    import test_score as TS
    import test_score_mot as TM
    A = TS.plane(0, 0, 0.0, 50, 50); B = TS.plane(3, 0, 0.0, 50, 50); far = TS.plane(40, 40, 0.0, 20, 25)
    part75 = np.concatenate([A[:1500], far])
    g2 = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A), dict(oid=2, first=0, last=100, pts=B)])
    g1 = TS.FakeGT([dict(oid=1, first=0, last=200, pts=A)])
    H = TS.plane(0, 3, 0.0, 50, 50); far2 = TS.plane(40, 40, 0.0, 50, 50)
    g3 = TS.FakeGT([dict(oid=1, first=0, last=100, pts=A), dict(oid=2, first=0, last=100, pts=B),
                    dict(oid=3, first=0, last=100, pts=H, color='23d5ea')])
    fr10 = [10 * (i + 1) for i in range(10)]
    cases = [
        ('C1 놓친 뒤 연속성', g2, [(10, [(1, A), (9, B)]), (20, [(1, far), (9, B)]), (30, [(1, part75), (2, A), (9, B)])],
         TM.vis_all(2, [10, 20, 30])),
        ('C2 don\'t care keyframe', g2, [(10, [(1, A), (9, B)]), (20, [(1, part75), (2, A), (9, B)]), (30, [(1, part75), (2, A), (9, B)])],
         dict(ep_index=np.array([0, 0, 0, 1, 1, 1]), frame=np.array([10, 20, 30] * 2), px_crop=np.array([5000, 800, 5000, 5000, 5000, 5000]))),
        ('C3 중복 관측 IDF1·HOTA', g1, [(f, ([(1, A)] if i < 5 else []) + [(2, part75)]) for i, f in enumerate(fr10)],
         TM.vis_all(1, fr10)),
        ('C4 없음 keyframe Frag', g2, [(f, [(1, A), (9, B)]) for f in (10, 20, 30, 40)],
         dict(ep_index=np.array([0] * 4 + [1] * 4), frame=np.array([10, 20, 30, 40] * 2), px_crop=np.array([5000, 5000, 800, 5000] + [5000] * 4))),
        ('C11 병합 관측 FP', g2, [(10, [(1, np.concatenate([A[:1100], B[:1100], far[:300]]))])], TM.vis_all(2, [10])),
        ('검출 없는 timestep', g1, [(10, [(1, part75)]), (20, [(1, far)]), (30, [(1, part75), (2, A)])], TM.vis_all(1, [10, 20, 30])),
        ('무시 GT 매칭 (present 아님 · 사람)', g3, [(10, [(1, A), (2, B), (3, H), (4, np.concatenate([A[:1250], far2[:1250]]))]),
                                               (20, [(1, A), (2, B), (3, H)])],
         dict(ep_index=np.array([0, 1, 2, 0, 1, 2]), frame=np.array([10, 10, 10, 20, 20, 20]),
              px_crop=np.array([5000, 800, 5000, 5000, 5000, 5000]))),
        ('작은 순수 조각 (IoU 0.26)', g1, [(f, [(1, A[:200]), (2, A)] if i % 3 == 0 else [(1, A[:200])]) for i, f in enumerate(fr10)],
         TM.vis_all(1, fr10)),
    ]
    for ci, (name, gt, kfs, vis) in enumerate(cases):
        TM.e2e(tmp, f'par_{ci}', gt, kfs, vis)
        d = tmp / f'par_{ci}'
        inp = SM.load_inputs(d, d / 'gt.h5', d / 'vis.npz')
        for unit in ('episode', 'object'):
            compare(f'{name} [{unit}]', TE, SM.sequence_data(inp, unit), inp['min_frac'])
            preproc_parity(f'{name} [{unit}]', TE, inp, unit)


def random_data(seed):
    """무작위 timestep 입력. 절반은 '진짜 짝 + 조각 + 잡음' 구조(연속성·IDSW 가 자주 걸리게), 절반은 완전 무작위."""
    rng = np.random.default_rng(seed)
    T = int(rng.integers(3, 40)); G = int(rng.integers(1, 7)); P = int(rng.integers(1, 12))
    structured = seed % 2 == 0
    owner = {g: int(rng.integers(P)) for g in range(G)}
    steps = []
    for t in range(T):
        if rng.random() < 0.1:
            steps.append(([], [], np.zeros((0, 0)))); continue
        gk = sorted(int(x) for x in rng.choice(G, size=int(rng.integers(0, G + 1)), replace=False))
        if structured:
            for g in gk:
                if rng.random() < 0.15:
                    owner[g] = int(rng.integers(P))
            tk = sorted({owner[g] for g in gk if rng.random() < 0.8} | {int(x) for x in rng.choice(P, size=int(rng.integers(0, 3)))})
            if rng.random() < 0.1:
                tk = []
            sim = np.zeros((len(gk), len(tk)))
            for i, g in enumerate(gk):
                for j, p in enumerate(tk):
                    if p == owner[g]:
                        sim[i, j] = rng.choice([0.5, rng.uniform(0.5, 1.0)])
                    elif rng.random() < 0.3:
                        sim[i, j] = rng.uniform(0.3, 0.9)
        else:
            tk = sorted(int(x) for x in rng.choice(P, size=int(rng.integers(0, P + 1)), replace=False))
            sim = rng.random((len(gk), len(tk)))
            sim[rng.random(sim.shape) < 0.4] = 0
            sim[rng.random(sim.shape) < 0.05] = 0.5
        steps.append((gk, tk, sim))
    return SM.data_from_timesteps(steps)


def random_cases(TE, n=60):
    print(f'2. 무작위 입력 {n}개')
    tested = 0
    for seed in range(n * 3):
        d = random_data(seed)
        if d['num_gt_dets'] == 0 or d['num_tracker_dets'] == 0:
            continue
        compare(f'seed {seed} (T={d["num_timesteps"]}, GT {d["num_gt_dets"]}, 예측 {d["num_tracker_dets"]})', TE, d)
        tested += 1
        if tested == n:
            break
    print('  검출 0개 경계 (TrackEval early return)')
    d = SM.data_from_timesteps([([1, 2], [], np.zeros((2, 0))), ([1], [], np.zeros((1, 0)))])
    compare('예측 검출 0', TE, d)


def real_score_dir(seq, tmp):
    col = 'gt_iou@20'
    cands = [os.environ.get('PARITY_SCORE_DIR'), run_dir(seq), REAL_RUNS / '_audit/fix_D_match3d' / seq]
    for c in cands:
        if c and (Path(c) / 'score_observations.csv').exists() and (Path(c) / 'score_kf_gt.csv').exists():
            with open(Path(c) / 'score_observations.csv') as fh:
                if col in fh.readline():
                    return Path(c)
    import test_score as TS
    labels = TS.real_labels_dir(seq)
    if labels is None or not (run_dir(seq) / 'frontend_output.h5').exists():
        return None
    print(f'  새 형식 score_observations.csv 없음 → 임시 폴더에 다시 채점 ({seq}, 라벨 {labels})')
    import score_frontend as SF
    out = tmp / f'real_{seq}'
    SF.score(run_dir(seq), seq_dir(seq), gt_h5(seq), vis=vis_npz(seq), out_dir=out, labels_dir=labels, log=lambda *a: None)
    return out


def real_case(TE, tmp, seq='apartment_s1_00h'):
    print(f'3. 실제 {seq}')
    if not (run_dir(seq) / 'frontend_output.h5').exists() or not gt_h5(seq).exists():
        print(f'  SKIP 실제 {seq}: frontend_output.h5 또는 GT h5 없음 (REAL_RUNS={REAL_RUNS})')
        return {}
    sd = real_score_dir(seq, tmp)
    if sd is None:
        print(f'  SKIP 실제 {seq}: 새 형식 채점 결과도 최신 버전 GT 라벨·frontend_output.h5 도 없음 (REAL_RUNS={REAL_RUNS})')
        return {}
    print(f'  채점 결과 폴더: {sd}')
    inp = SM.load_inputs(run_dir(seq), gt_h5(seq), vis_npz(seq), score_dir=sd)
    tables = {}
    for unit in ('episode', 'object'):
        d = SM.sequence_data(inp, unit)
        tables[unit] = compare(f'{seq} [{unit}] (GT 검출 {d["num_gt_dets"]}, 예측 검출 {d["num_tracker_dets"]})', TE, d,
                               inp['min_frac'], show=True)
        preproc_parity(f'{seq} [{unit}]', TE, inp, unit)
    return tables


if __name__ == '__main__':
    TE = import_trackeval()
    if TE is None:
        sys.exit(0)
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        synthetic_e2e(TE, tmp)
        random_cases(TE)
        if '--no-real' not in sys.argv:
            tables = real_case(TE, tmp)
            if os.environ.get('PARITY_JSON'):
                Path(os.environ['PARITY_JSON']).write_text(json.dumps(tables, indent=1, ensure_ascii=False))
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)
