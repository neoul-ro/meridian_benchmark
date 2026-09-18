#!/usr/bin/env python3
"""match3d.py 자체 검증 — 3D 매칭 규칙(허용 거리 IoU_τ · 매칭 · 미매칭 예측 분류 · GT 상태 · 트랙 원인)을 손계산 입력으로 확인한다.

격자: 모든 점은 2cm 셀 중심. cells(i0, j0, k0, nx, ny) = x 셀 i0..i0+nx-1 · y 셀 j0..j0+ny-1 · z 셀 k0 (y 바깥 루프).
τ = 20cm 는 '셀 거리 < 10' (strict). 같은 행에서 열 c 의 GT 셀은 가장 가까운 예측 열과의 차이가 9 이하일 때만 τ 이내.

M1 공식          IoU_τ = P·R / (P + R − P·R) = F/(2−F) (Jaccard–F1 항등식)
M2 τ→0 극한      같은 표본 밀도에서 exact IoU |A∩B|/|A∪B| 와 같다
M3 경계          정확히 τ(10셀 · 대각 (6,8,0) 셀)은 τ 밖 · 9셀은 안 · float32 GT 좌표(x=10.03m)에서도 같다 · τmax=50cm 에서 48cm 는 안
M4 조각 사례     GT 100셀 선분 · 작은 조각(앞 10셀, P=1) vs 큰 조각(85셀 + GT 밖 15셀, P=0.85)
                 share 최대(옛 방식)는 작은 조각, IoU 최대는 큰 조각 (IoU 0.19 vs 0.806256)
M5 매칭 형식     TrackEval clear.py:82 'sim < thr − eps → 0' · :86 '> 0 + eps' · 헝가리안 합 최대
M6 FP / 제외     panopticapi evaluation.py:161 — (void + 무시 GT 위 복셀) / |P| > 0.5 이면 제외. 정확히 0.5 는 FP
M7 GT 상태       merged · split · low_iou(0.48 · 0.25 · 0.24 — Loc 하한 0.1 이상) · missed(IoU 0)
                 [F2] Loc 하한 = TIDE 기본 background_threshold 0.1 (tidecv 1.0.1 quantify.py:428)
M8 중복 [E2]     TIDE Dupe (quantify.py:249-255): 매칭 안 된 예측이 다른 예측에 매칭된 GT 와 IoU_τ ≥ 0.5 → dup.
                 작은 조각(IoU 0.215, P_τ 는 1)은 옛 규칙(P_τ 최대 GT)으로는 dup 이었지만 TIDE 로는 Bkg
M9 트랙 원인     present keyframe 0 → no_kf · 최빈값 · 동률은 merged > split > low_iou > missed
M10 CSV 값 형식  'ep:iou;…' 17 유효숫자 — float64 왕복이 정확 (score_mot 가 같은 유사도를 읽는다)
M11 가지치기 검증 pair_stats(최근접 거름 · bbox · 부분 조회)가 전체 거리 행렬로 센 nP·nR·near·void·cover 와 같다 (무작위 겹친 점군 20판)
                 [E1] 배타적 덮음 nRx · xcover 도 전체 거리 행렬로 센 값과 같다
M12 배타적 덮음 [E1] 벽 예측이 20cm 안의 얇은 문틀 GT 를 '덮었다' 고 세지 않는다 (merged 아님) · 벽 조각들이 문틀의 split 조각이 아니다 ·
                 진짜로 두 GT 를 합친 예측은 여전히 merged · 같은 거리 동률은 조건을 만족하는 예측 모두가 덮음 (순서 무관)
M13 TIDE 규칙 [E2·E3] tide_rules.pred_error — quantify.py:229-265 순서 (Bkg(GT 없음) · Loc [bg, pos] · Dupe · Bkg · Other),
                 무시(present 아님) GT 는 판정에서 뺌 (quantify.py:18) · Loc 경계 LOC_MIN_IOU 는 한 곳(tide_rules)에서만
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILS = []
V = 0.02


def check(name, got, want, tol=None):
    try:
        if tol is None:
            ok = bool(np.all(np.asarray(got) == np.asarray(want))) if isinstance(want, (list, tuple, np.ndarray)) else got == want
        else:
            ok = got is not None and bool(np.all(np.abs(np.asarray(got, float) - np.asarray(want, float)) <= tol))
    except Exception:  # noqa: BLE001
        ok = False
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got} want={want}')
    if not ok:
        FAILS.append(name)


def cells(i0, j0, k0, nx, ny):
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack([(i0 + ix.ravel() + 0.5) * V, (j0 + iy.ravel() + 0.5) * V, np.full(ix.size, (k0 + 0.5) * V)], 1)


def _M():
    import match3d as M  # noqa: E402  (고치기 전 코드에는 없는 모듈 — 각 테스트가 실패로 센다)
    return M


def test_formula():
    """M1 공식"""
    M = _M()
    check('IoU(P=0.75, R=0.6) = 0.45/0.9 = 0.5', float(M.tolerant_iou(0.75, 0.6)), 0.5, 1e-12)
    check('IoU(P=0.85, R=0.94) = 0.799/0.991', float(M.tolerant_iou(0.85, 0.94)), 0.799 / 0.991, 1e-12)
    check('IoU(1, 0) = 0', float(M.tolerant_iou(1.0, 0.0)), 0.0)
    check('IoU(0, 0) = 0 (0/0 가드)', float(M.tolerant_iou(0.0, 0.0)), 0.0)
    rng = np.random.default_rng(0)
    P, R = rng.random(50), rng.random(50)
    F = 2 * P * R / (P + R)
    check('Jaccard–F1 항등식 J = F/(2−F) (무작위 50쌍)', M.tolerant_iou(P, R), F / (2 - F), 1e-12)


def test_exact_limit():
    """M2 τ→0 에서 exact IoU"""
    M = _M()
    A = cells(0, 0, 0, 10, 1); B = cells(5, 0, 0, 10, 1)            # 겹침 5셀 · 합집합 15셀
    st = M.pair_stats([A], [B], taus=(0.001,), main_tau=0.001)
    check('nP = |A∩B| = 5', int(st.nP[0, 0, 0]), 5); check('nR = 5', int(st.nR[0, 0, 0]), 5)
    check('IoU_τ(τ→0) = 5/15', float(st.iou(0)[0, 0]), 5 / 15, 1e-12)


def test_strict_boundary():
    """M3 strict τ 경계"""
    M = _M()
    G = cells(0, 0, 0, 1, 1)
    preds = [cells(0, 0, 10, 1, 1),      # z 10셀 = 20cm
             cells(0, 0, 9, 1, 1),       # 18cm
             cells(6, 8, 0, 1, 1),       # 대각 √(36+64) = 10셀 = 20cm
             cells(0, 0, 24, 1, 1),      # 48cm
             cells(0, 0, 25, 1, 1)]      # 50cm
    taus = (0.10, 0.20, 0.50)
    st = M.pair_stats(preds, [G], taus=taus, main_tau=0.20)
    check('τ20: 20cm·대각 20cm 는 밖, 18cm 만 안', st.nP[0, :, 1], [0, 1, 0, 0, 0])
    check('τ20 R 도 대칭', st.nR[0, :, 1], [0, 1, 0, 0, 0])
    check('τ50: 48cm 안 · 50cm 밖 (KD-tree 상한 = τmax)', st.nP[0, :, 2], [1, 1, 1, 1, 0])
    check('τ10: 전부 밖', st.nP[0, :, 0], [0, 0, 0, 0, 0])
    check('가장 가까운 GT 라벨 수 (τ20) = 18cm 만', st.near[0], [0, 1, 0, 0, 0])
    check('void (τ20) = 나머지', st.void, [1, 0, 1, 1, 1])
    # float32 GT (실제 h5 · 감사 A-6 의 x=10.03m): 반올림 오차 ~2e-6m 가 있어도 20cm 는 밖, 18cm 는 안
    ys, zs = np.meshgrid(np.arange(30), np.arange(30))
    g64 = np.stack([np.full(ys.size, (501 + 0.5) * V), (ys.ravel() + 0.5) * V, (zs.ravel() + 0.5) * V], 1)
    st = M.pair_stats([g64 + [0.20, 0, 0], g64 + [0.18, 0, 0]], [g64.astype(np.float32)], taus=(0.2,), main_tau=0.2)
    check('float32 GT: +20cm 복셀 0개 τ 이내', int(st.nP[0, 0, 0]), 0)
    check('float32 GT: +18cm 복셀 900개 τ 이내', int(st.nP[0, 1, 0]), 900)


def test_fragment():
    """M4 조각 사례 — share 최대 vs IoU 최대"""
    M = _M()
    G = cells(0, 0, 0, 100, 1)
    small = G[:10]
    big = np.concatenate([G[:85], cells(0, 50, 0, 15, 1)])            # GT 밖 15셀은 1m 떨어진 행
    st = M.pair_stats([small, big], [G], taus=(0.2,), main_tau=0.2)
    share = st.nP[0, :, 0] / st.n_pred
    check('share (옛 기준) = [1.0, 0.85] → 옛 헝가리안은 작은 조각', share, [1.0, 0.85], 1e-12)
    # small: P=1, R = 열 0..18 = 19/100 → IoU 0.19 · big: P=0.85, R = 열 0..93 = 0.94 → 0.799/0.991
    check('IoU = [0.19, 0.806256]', st.iou(0)[0], [0.19, 0.799 / 0.991], 1e-9)
    rows, cols = M.match(st.iou(0))
    check('IoU 매칭 = 큰 조각 (열 1)', (rows.tolist(), cols.tolist()), ([0], [1]))


def test_match_form():
    """M5 매칭 형식"""
    M = _M()
    r, c = M.match(np.array([[0.5]]))
    check('정확히 0.5 → 매칭 (≥ thr − eps)', (r.tolist(), c.tolist()), ([0], [0]))
    r, c = M.match(np.array([[0.5 - 1e-17]]))
    check('0.5 − 1e-17 (float 로 0.5) → 매칭', len(r), 1)
    r, c = M.match(np.array([[0.4999999]]))
    check('0.4999999 → 매칭 안 됨', len(r), 0)
    r, c = M.match(np.array([[0.9, 0.6], [0.8, 0.0]]))            # 합 최대: 0.6 + 0.8 = 1.4 > 0.9
    check('합 최대 = (GT0,예측1)·(GT1,예측0)', sorted(zip(r.tolist(), c.tolist())), [(0, 1), (1, 0)])
    r, c = M.match(np.zeros((0, 3)))
    check('빈 행렬', len(r), 0)


def test_unmatched_class():
    """M6 FP / 제외 경계 (panopticapi evaluation.py:161 '> 0.5')"""
    M = _M()
    F, T = False, True
    check('void 50/100 → FP', M.unmatched_class(100, 50, np.array([50]), np.array([F])), ('fp', ''))
    check('void 51/100 → 제외 (void)', M.unmatched_class(100, 51, np.array([49]), np.array([F])), ('excluded', 'void'))
    check('void 30 + 무시 GT 30 → 제외 (void+ignored_gt)',
          M.unmatched_class(100, 30, np.array([40, 30]), np.array([F, T])), ('excluded', 'void+ignored_gt'))
    check('있는 GT 50 + 무시 GT 50 → FP', M.unmatched_class(100, 0, np.array([50, 50]), np.array([F, T])), ('fp', ''))
    check('void 40 + 무시 GT 60 → 제외 (ignored_gt)',
          M.unmatched_class(100, 40, np.array([0, 60]), np.array([F, T])), ('excluded', 'ignored_gt'))
    check('복셀 0 → 제외 (no_voxels_within_range)', M.unmatched_class(0, 0, np.array([0]), np.array([F])),
          ('excluded', 'no_voxels_within_range'))


def test_gt_status():
    """M7 GT 상태 — score_2d.py 규칙을 τ 허용 복셀 집합에 그대로"""
    M = _M()
    M1 = cells(0, 0, 0, 30, 30); M2 = cells(50, 0, 0, 30, 30)       # 21셀 떨어짐
    S = cells(0, 100, 0, 90, 10)
    L48 = cells(0, 200, 0, 50, 10); L25 = cells(0, 300, 0, 40, 10); L24 = cells(0, 400, 0, 50, 10)
    E = cells(0, 500, 0, 20, 20); T = cells(200, 0, 0, 20, 20)
    gts = [M1, M2, S, L48, L25, L24, E, T]
    preds = [np.concatenate([M1, M2, cells(0, 1000, 0, 100, 1)]),   # 0: M1·M2 를 한 덩어리로 (+ void 100) → P = 900/1900
             cells(0, 100, 0, 20, 10), cells(35, 100, 0, 20, 10), cells(70, 100, 0, 20, 10),   # 1-3: S 의 세 조각
             cells(0, 200, 0, 15, 10),                               # 4: L48 앞 15열 → R = 24/50
             cells(0, 300, 0, 1, 10),                                # 5: L25 첫 열 → R = 10/40 = 0.25
             cells(0, 400, 0, 3, 10),                                # 6: L24 앞 3열 → R = 12/50 = 0.24
             T.copy()]                                               # 7: T 정확히
    st = M.pair_stats(preds, gts, taus=(0.2,), main_tau=0.2)
    iou = st.iou(0)
    check('IoU(M1, 덩어리) = 900/1900 (< 0.5)', float(iou[0, 0]), 900 / 1900, 1e-12)
    check('IoU(S, 조각) = [29/90, 38/90, 29/90]', iou[2, 1:4], [29 / 90, 38 / 90, 29 / 90], 1e-12)
    check('IoU(L25, 첫 열) = 0.25', float(iou[4, 5]), 0.25, 1e-12)
    rows, cols = M.match(iou)
    check('매칭은 T 하나', (rows.tolist(), cols.tolist()), ([7], [7]))
    stat = M.gt_statuses(st, rows, cols)
    check('상태 (L24 는 0.24 ≥ 0.1 이라 low_iou, F2)', [s['status'] for s in stat],
          ['merged', 'merged', 'split', 'low_iou', 'low_iou', 'low_iou', 'missed', 'tp'])
    check('merged 를 만든 예측 = 0', (stat[0]['merged_pred'], stat[1]['merged_pred']), (0, 0))
    check('split 조각 3 · 합집합 덮음 1.0', (stat[2]['n_frags'], stat[2]['split_cover']), (3, 1.0))
    check('split 합집합 복셀 수 = 900', st.cover(2, [1, 2, 3]), 900)
    check('best IoU (L48 · L24 · E)', [stat[3]['best_iou'], stat[5]['best_iou'], stat[6]['best_iou']], [0.48, 0.24, 0.0], 1e-12)
    check('best_pred (L48 = 4 · E 없음 = -1)', (stat[3]['best_pred'], stat[6]['best_pred']), (4, -1))


def test_duplicate():
    """M8 [E2] 중복 = TIDE Dupe — 다른 예측에 매칭된 GT 와 IoU_τ ≥ 0.5 (quantify.py:249-255)"""
    M = _M()
    G1 = cells(0, 0, 0, 20, 20); G2 = cells(100, 0, 0, 20, 20)
    preds = [G1.copy(),                  # 0: G1 매칭
             cells(0, 0, 0, 20, 4),      # 1: G1 0..3행 → P 1 · R = 0..12행 = 260/400 → IoU 0.65 ≥ 0.5 → Dupe
             cells(0, 0, 0, 1, 1),       # 2: G1 모서리 1복셀 → P 1 · R = 사분원 86/400 → IoU 0.215 → Bkg (옛 규칙: P_τ 최대 G1 → dup)
             cells(100, 0, 0, 2, 2),     # 3: 매칭 안 된 G2 모서리 2×2 → R 107/400 = 0.2675 → Loc, 중복 아님
             cells(0, 900, 0, 5, 5)]     # 4: GT 없음 → Bkg
    st = M.pair_stats(preds, [G1, G2], taus=(0.2,), main_tau=0.2)
    iou = st.iou(0)
    check('IoU (G1, 예측1 · 예측2) = 0.65 · 0.215', [float(iou[0, 1]), float(iou[0, 2])], [0.65, 86 / 400], 1e-12)
    check('IoU (G2, 예측3) = 107/400', float(iou[1, 3]), 107 / 400, 1e-12)
    rows, cols = M.match(iou)
    check('매칭 = (G1, 예측0) 하나', (rows.tolist(), cols.tolist()), ([0], [0]))
    check('예측1 (IoU 0.65, G1 은 예측0 이 가짐) → dup G1', M.duplicate_of(st, 1, rows, cols), 0)
    check('예측2 (IoU 0.215 작은 조각) → 중복 아님 [E2: 옛 규칙은 dup]', M.duplicate_of(st, 2, rows, cols), -1)
    check('예측3 (매칭 안 된 G2 위) → 중복 아님', M.duplicate_of(st, 3, rows, cols), -1)
    check('예측4 (GT 없음) → 중복 아님', M.duplicate_of(st, 4, rows, cols), -1)
    check('매칭된 예측0 → 중복 아님', M.duplicate_of(st, 0, rows, cols), -1)
    errs = [M.pred_error(st, i, rows, cols) for i in range(1, 5)]
    check('TIDE 오류 (예측1..4) = dupe · loc(0.215 ≥ 0.1) · loc · bkg', [e[0] for e in errs], ['dupe', 'loc', 'loc', 'bkg'])
    check('dupe 의 GT = G1(0) · loc 의 GT = G1(0) · G2(1)', (errs[0][1], errs[1][1], errs[2][1]), (0, 0, 1))
    valid = np.array([False, True])                                       # G1 이 present 아님(무시 GT) → TIDE 판정에서 빠진다
    check('G1 이 무시 GT 면 예측1 은 Dupe 아님 (quantify.py:18 ex.gt 는 무시 GT 제외) → bkg',
          M.pred_error(st, 1, rows, cols, valid=valid)[0], 'bkg')
    check('duplicate_of 도 같은 규칙', M.duplicate_of(st, 1, rows, cols, valid=valid), -1)


def test_track_reason():
    """M9 트랙 원인"""
    M = _M()
    check('present keyframe 0 → no_kf', M.track_reason(0, []), 'no_kf')
    check('최빈 low_iou', M.track_reason(3, ['low_iou', 'low_iou', 'missed']), 'low_iou')
    check('동률 merged·missed → merged', M.track_reason(2, ['missed', 'merged']), 'merged')
    check('동률 low_iou·split → split', M.track_reason(2, ['low_iou', 'split']), 'split')
    check('동률 missed·low_iou (2:2) → low_iou', M.track_reason(4, ['missed', 'missed', 'low_iou', 'low_iou']), 'low_iou')
    check('옛 값 대응표', M.OLD_REASON_MAP, {'absorbed': 'merged', 'not_detected': 'missed'})


def test_values_format():
    """M10 CSV 값 형식"""
    M = _M()
    d = {3: 13 / 21, 0: 1.0, 7: 0.1 + 0.2}
    s = M.format_values(d)
    check('ep 오름차순', s.split(';')[0].split(':')[0], '0')
    back = M.parse_values(s)
    check('float64 왕복 정확', [back[k] == d[k] for k in sorted(d)], [True, True, True])
    check('빈 문자열 → {}', M.parse_values(''), {})


def test_bruteforce():
    """M11 pair_stats = 전체 거리 행렬 (무작위 20판)"""
    M = _M()
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(1)
    taus = (0.10, 0.20, 0.50)
    bad = 0
    for trial in range(20):
        def blob():
            c = rng.integers(-40, 40, 3); n = int(rng.integers(1, 400))
            k = np.unique(c + rng.integers(-12, 13, (n, 3)) * np.array([1, 1, int(rng.integers(0, 2))]), axis=0)
            return (k + 0.5) * V
        preds = [blob() for _ in range(int(rng.integers(1, 7)))] + [np.zeros((0, 3))]
        gts = [blob() for _ in range(int(rng.integers(1, 7)))]
        st = M.pair_stats(preds, gts, taus, 0.20)
        for j, g in enumerate(gts):
            for i, p in enumerate(preds):
                if not len(p):
                    ok = not st.nP[j, i].any() and not st.nR[j, i].any()
                else:
                    D = cdist(p, g)
                    ok = all(st.nP[j, i, t] == int((D.min(1) < tau - M.EPS).sum()) and
                             st.nR[j, i, t] == int((D.min(0) < tau - M.EPS).sum()) for t, tau in enumerate(taus))
                bad += int(not ok)
        for i, p in enumerate(preds):
            if not len(p):
                continue
            Dg = np.stack([cdist(p, g).min(1) for g in gts], 1)          # 복셀 × GT 최근접 거리
            dmin = Dg.min(1); inm = dmin < 0.20 - M.EPS
            jmin = np.argmax(Dg <= dmin[:, None] + 1e-9, axis=1)        # 규칙: 같은 거리(±1e-9)면 번호가 작은 GT
            near = np.bincount(jmin[inm], minlength=len(gts))
            bad += int(not (np.array_equal(near, st.near[:, i]) and st.void[i] == len(p) - inm.sum()))
        for j, g in enumerate(gts):
            frag = [i for i in range(len(preds)) if len(preds[i])]
            if frag:
                cov = int((cdist(g, np.concatenate([preds[i] for i in frag])).min(1) < 0.20 - M.EPS).sum())
                bad += int(st.cover(j, frag) != cov)
        # [E1] 배타적 덮음: GT 복셀 y 는 예측 p 가 덮음 ⇔ y 의 최근접 예측 복셀(거리 < τ, 동률 ±1e-9 는 모두 후보) 중
        #      p 의 것이 있고 그 복셀의 가장 가까운 GT 가 이 GT
        nz = [i for i in range(len(preds)) if len(preds[i])]
        if nz:
            owner = {}
            for i in nz:
                Dg = np.stack([cdist(preds[i], g).min(1) for g in gts], 1)
                dmin = Dg.min(1); inm = dmin < 0.20 - M.EPS
                owner[i] = np.where(inm, np.argmax(Dg <= dmin[:, None] + 1e-9, axis=1), -1)
            for j, g in enumerate(gts):
                Dp = {i: cdist(g, preds[i]) for i in nz}
                dall = np.min(np.stack([Dp[i].min(1) for i in nz], 1), 1)
                cov_by = {}
                for i in nz:
                    own = Dp[i][:, owner[i] == j]
                    dn = own.min(1) if own.shape[1] else np.full(len(g), np.inf)
                    cov_by[i] = (dn < 0.20 - M.EPS) & (dn <= dall + 1e-9)
                    bad += int(st.nRx[j, i] != int(cov_by[i].sum()))
                bad += int(st.xcover(j, nz) != int(np.any(np.stack([cov_by[i] for i in nz]), 0).sum()))
                half = nz[:max(1, len(nz) // 2)]
                bad += int(st.xcover(j, half) != int(np.any(np.stack([cov_by[i] for i in half]), 0).sum()))
    check('전체 거리 행렬과 다른 칸 수 (nP·nR·near·void·cover · [E1] nRx·xcover, 20판)', bad, 0)


def test_exclusive_coverage():
    """M12 [E1] merged·split 의 덮음은 배타적 (가장 가까운 GT 소유)"""
    M = _M()
    W = cells(0, 0, 0, 60, 60)                          # 벽 GT (z 셀 0) 3600
    Tr = cells(0, 60, 5, 60, 3)                         # 문틀 GT: 벽 바로 옆 3행, 10cm 위 180 — 벽 복셀 20cm 안
    st = M.pair_stats([W.copy()], [W, Tr], taus=(0.2,), main_tau=0.2)
    check('비배타 R(문틀, 벽 예측) = 1 (20cm 허용 때문)', int(st.nR[1, 0, 0]), 180)
    check('배타 nRx: 벽 3600 · 문틀 0 (최근접 예측 복셀의 주인이 벽)', [int(st.nRx[0, 0]), int(st.nRx[1, 0])], [3600, 0])
    rows, cols = M.match(st.iou(0))
    stat = M.gt_statuses(st, rows, cols)
    check('벽 예측이 벽에 매칭 · 문틀은 merged 가 아님 (옛: merged) · 최고 IoU 0.133 은 Loc', [s['status'] for s in stat], ['tp', 'low_iou'])
    # 진짜 합친 예측 (벽 ∪ 문틀): 문틀 복셀이 예측에 그대로 있다 → 여전히 merged
    st = M.pair_stats([np.concatenate([W, Tr])], [W, Tr], taus=(0.2,), main_tau=0.2)
    rows, cols = M.match(st.iou(0))
    stat = M.gt_statuses(st, rows, cols)
    check('벽 ∪ 문틀 예측: nRx = [3600, 180]', [int(st.nRx[0, 0]), int(st.nRx[1, 0])], [3600, 180])
    check('벽 ∪ 문틀 예측: 벽 tp · 문틀 merged', [s['status'] for s in stat], ['tp', 'merged'])
    # split: 벽 끝줄 5행을 10칸씩 6조각 — 각 조각의 복셀은 전부 문틀 20cm 안(비배타 P = 1)이지만 주인은 벽
    pieces = [cells(10 * k, 55, 0, 10, 5) for k in range(6)]
    st = M.pair_stats(pieces, [W, Tr], taus=(0.2,), main_tau=0.2)
    rows, cols = M.match(st.iou(0))
    check('조각은 아무것도 매칭 안 됨 (최대 IoU < 0.5)', len(rows), 0)
    check('비배타 P(조각, 문틀) = 1 (6조각 모두)', [int(st.nP[1, i, 0]) for i in range(6)], [50] * 6)
    check('배타 P: near(문틀) = 0 (6조각 모두)', [int(st.near[1, i]) for i in range(6)], [0] * 6)
    stat = M.gt_statuses(st, rows, cols)
    check('문틀 = low_iou (옛: 조각 6 · 합집합 1.0 → split)', (stat[1]['status'], stat[1]['n_frags']), ('low_iou', 0))
    check('문틀 best IoU = 26/60 (가운데 조각)', stat[1]['best_iou'], 26 / 60, 1e-12)
    # 평행한 두 GT 표면(10cm 간격) 위의 예측 하나: 아래 GT 에 매칭, 위 GT 는 IoU_τ 1.0 이지만 예측 복셀의 주인이 아래 GT
    #   → 배타 덮음 0 → merged 아님. 최고 IoU 1.0 > 0.5 라 TIDE Loc 범위 [0.1, 0.5] 밖 → missed (옛: merged)
    A = cells(0, 0, 0, 30, 30); B = cells(0, 0, 5, 30, 30)
    st = M.pair_stats([A.copy()], [A, B], taus=(0.2,), main_tau=0.2)
    stat = M.gt_statuses(st, *M.match(st.iou(0)))
    check('평행 표면: IoU_τ (A, B) = (1, 1) · nRx = (900, 0)', ([float(x) for x in st.iou(0)[:, 0]], [int(x) for x in st.nRx[:, 0]]),
          ([1.0, 1.0], [900, 0]))
    check('평행 표면: A tp · B missed (IoU 1.0 은 Loc 범위 밖, quantify.py:237 ≤ pos)', [s_['status'] for s_ in stat], ['tp', 'missed'])
    # 동률: 똑같은 예측 둘이 두 GT 를 합침 → 두 예측 모두 덮음 (작은 번호에만 주지 않는다)
    A = cells(0, 0, 0, 30, 30); B = cells(50, 0, 0, 30, 30)
    blob = np.concatenate([A, B, cells(0, 1000, 0, 100, 1)])
    st = M.pair_stats([blob, blob.copy()], [A, B], taus=(0.2,), main_tau=0.2)
    check('동률: nRx(A) = [900, 900]', [int(st.nRx[0, 0]), int(st.nRx[0, 1])], [900, 900])
    stat = M.gt_statuses(st, *M.match(st.iou(0)))
    check('동률: A · B 모두 merged', [s['status'] for s in stat], ['merged', 'merged'])


def test_tide_rule():
    """M13 [E2·E3] TIDE 오류 판정 순서 (quantify.py:229-265) · Loc 경계 한 곳"""
    import tide_rules as TR
    pe = lambda iou, used, valid=None: TR.pred_error(np.array(iou, float), np.array(used, bool),
                                                     None if valid is None else np.array(valid, bool), pos=0.5)
    check('GT 없음 → bkg (quantify.py:230-233)', pe([], []), ('bkg', -1))
    check('0.30 → loc', pe([0.30], [False]), ('loc', 0))
    check('정확히 0.1 → loc (quantify.py:237 bg ≤ IoU, bg 기본 0.1 = :428)', pe([0.1], [False]), ('loc', 0))
    check('0.0999 → bkg (quantify.py:259)', pe([0.0999], [False]), ('bkg', -1))
    check('0.24 → loc (옛 하한 0.25 에서는 bkg)', pe([0.24], [False]), ('loc', 0))
    check('사용된 GT 와 정확히 0.5 → loc (:237 IoU ≤ pos 가 Dupe 보다 먼저)', pe([0.5], [True]), ('loc', 0))
    check('사용된 GT 와 0.6 → dupe', pe([0.1, 0.6], [False, True]), ('dupe', 1))
    check('사용된 GT 0.6 · 안 쓴 GT 0.4 → dupe (최대 0.6 은 Loc 범위 밖)', pe([0.4, 0.6], [False, True]), ('dupe', 1))
    check('안 쓴 GT 와 0.6 → other (:264-265)', pe([0.6], [False]), ('other', -1))
    check('무시 GT 와 0.9(사용됨) · 유효 GT 0.05 → bkg', pe([0.9, 0.05], [True, False], [False, True]), ('bkg', -1))
    check('유효 GT 가 모두 무시 → bkg (ex.gt 비어 있음)', pe([0.9], [True], [False]), ('bkg', -1))
    check('LOC_MIN_IOU = TIDE 기본 background_threshold 0.1', TR.LOC_MIN_IOU, 0.1)
    check('is_loc 범위 [0.1, pos] 양쪽 포함 (0.0999 · 0.1 · 0.5 · 0.5001)',
          [TR.is_loc(x, 0.5) for x in (0.0999, 0.1, 0.5, 0.5001)], [False, True, True, False])
    M = _M()
    old = TR.LOC_MIN_IOU
    try:
        TR.LOC_MIN_IOU = 0.3                                              # 한 곳만 바꾸면 3D 판정이 따라 바뀐다
        L25 = cells(0, 300, 0, 40, 10)
        st = M.pair_stats([cells(0, 300, 0, 1, 10)], [L25], taus=(0.2,), main_tau=0.2)
        check('LOC_MIN_IOU 0.3 이면 IoU 0.25 GT 는 missed (match3d 가 tide_rules 값을 읽음)',
              M.gt_statuses(st, *M.match(st.iou(0)))[0]['status'], 'missed')
    finally:
        TR.LOC_MIN_IOU = old
    check('되돌리면 low_iou', M.gt_statuses(st, *M.match(st.iou(0)))[0]['status'], 'low_iou')


def run():
    for fn in (test_bruteforce, test_formula, test_exact_limit, test_strict_boundary, test_fragment, test_match_form, test_unmatched_class,
               test_gt_status, test_duplicate, test_track_reason, test_values_format, test_exclusive_coverage, test_tide_rule):
        print(f'[{fn.__name__}] {fn.__doc__.strip()}')
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=3)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(f'{fn.__name__} 예외')
    return FAILS


if __name__ == '__main__':
    run()
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)
