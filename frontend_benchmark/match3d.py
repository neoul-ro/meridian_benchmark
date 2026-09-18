#!/usr/bin/env python3
"""3D 매칭 규칙 한 곳 — score_frontend.py(검출·원인)와 score_mot.py(추적 전처리·유사도)가 같이 쓴다 (26/09/17 사용자 결정 ④).

입력은 keyframe 하나의 복셀 집합들: 예측(관측) i 의 2cm 셀 중심 Pv_i, GT j 의 표면 셀 중심 Gv_j (gt_surface.LabelSurface).

허용 거리 IoU (tolerant IoU)
  P_τ(i, j) = |{x ∈ Pv_i : d(x, Gv_j) < τ}| / |Pv_i|          (예측 쪽 정밀도 · T&T evaluation.py:173-176 의 precision)
  R_τ(i, j) = |{y ∈ Gv_j : d(y, Pv_i) < τ}| / |Gv_j|          (GT 쪽 재현율 · 같은 곳의 recall)
  IoU_τ     = P·R / (P + R − P·R)
            = F/(2 − F), F = 2PR/(P+R) — Jaccard 와 F1 의 항등식 J = F/(2−F).
            τ → 0 이고 두 집합의 표본이 같은 격자면 P = |∩|/|Pv|, R = |∩|/|Gv| 라 식이 |∩|/|∪| (exact IoU) 가 된다.
            MOTS(마스크 IoU, TrackEval kitti_mots.py:390) · ScanNet 인스턴스 IoU(evaluate_semantic_instance.py:119
            'intersection / (gt + pred − intersection)') 와 같은 유사도를, 표본 격자 오차를 τ 로 흡수해서 3D 복셀에 쓴 것.
  거리 판정  d < τ − EPS (strict, T&T 와 같음). EPS = 1e-5m: 두 쪽 모두 float64 셀 중심이라 격자 거리는 √정수·v 이고 τ 바로
            안쪽 격자 거리까지 간격 ≥ v²/(2τ) ≈ 4e-4m (τ=50cm). GT 좌표가 float32(오차 ≤ 3.3e-6m, 감사 A-6)여도 판정이 안 바뀐다.
            KD-tree 는 distance_upper_bound = max(τ) (scipy 도 strict '<') 로 조회한다.

매칭 (keyframe 안)
  후보 = IoU_τ ≥ 0.5 인 (GT, 예측) 쌍, 헝가리안 IoU 합 최대. 행렬 모양·임계 처리는 TrackEval 그대로:
  clear.py:82 'score_mat[similarity < threshold − eps] = 0' · :85 linear_sum_assignment(−score) · :86 '> 0 + eps'
  (mot_challenge_2d_box.py:374-377 · kitti_mots.py:329-332 의 전처리 매칭도 같은 형식). 행 = GT, 열 = 예측.

미매칭 예측 — FP / 제외 (panopticapi evaluation.py:155-163 'ignored if more than half of the segment correspond to VOID and
  CROWD regions' · ScanNet evaluate_semantic_instance.py:152-163 · KITTI MOTS kitti_mots.py:336-344 무시 영역 50%)
  void 복셀     = τ 안에 어떤 GT 표면도 없는 예측 복셀
  무시 GT 위    = 가장 가까운 GT 표면 복셀(τ 이내)이 무시 GT(score_frontend: present 아님, score_mot: present 아님·사람)의 것
                 — 복셀 하나는 가장 가까운 GT 하나에만 속한다(픽셀 라벨이 하나인 panopticapi 와 같게 겹쳐 세지 않는다).
                 거리가 같으면(±1e-9m) GT 번호(= 에피소드 번호 순서)가 작은 쪽.
  (void + 무시 GT 위) / |P| > 0.5 → 제외 (정수 비교 2·n > |P|, 정확히 0.5 는 FP). 이유 = void / ignored_gt / void+ignored_gt
  (각 항목 하나만으로 절반을 넘으면 그 이름). 복셀 0개(5m 밖) → 제외 no_voxels_within_range.

present GT 의 미검출 원인 — score_2d.score_kf 의 merger · GT 상태 규칙을 τ 허용 복셀 집합에 옮긴 것 (판정 순서도 같다)
  덮음은 배타적 (수정 ⑤ E1). 매칭(IoU_τ)은 위의 비배타 P_τ · R_τ 그대로이고, merged · split 판정에만 아래를 쓴다.
  2D 에서는 픽셀 하나가 GT 라벨 하나만 가져서 '덮음' 이 저절로 배타적이다. 3D 에서 20cm 허용을 그대로 쓰면 큰 평면 예측 근처의
  얇은 GT(문틀 등)도 R_τ ≥ 0.5 가 되어 merged 가 부풀었다 (감사 fix_D fig/apartment_s1_00h_kf58_ep146_merged.png: 벽 예측 하나가
  문틀 띠 ep118·119·120 을 '덮음'). 그래서 FP/제외에 쓰는 '가장 가까운 GT 소유' (nearest-GT ownership) 를 덮음에도 쓴다:
    소유   예측 복셀 x 의 주인 = 주 τ 안에서 가장 가까운 GT (동률 ±1e-9m 는 번호 작은 GT) — near[j, i] 가 세는 것과 같다
    Rx(i, j) = |{y ∈ Gv_j : y 의 최근접 예측 복셀(거리 < τ) 중 예측 i 의 것이 있고, 그 복셀의 주인이 j}| / |Gv_j|
             최근접이 여러 개(같은 거리 ±1e-9m)면 조건을 만족하는 예측 모두가 덮는다 — 겹친 SAM 마스크가 한 픽셀을 함께 덮는
             2D 와 같다. 예측 번호 순서에 따라 달라지지 않는다.
    Px(i, j) = near[j, i] / |Pv_i|  (예측 복셀 중 주인이 j 인 비율 — 2D 의 '예측이 GT 안에 있는 비율')
  merged  (과소분할, Hoover 1996 under-segmentation) Rx(i, j) ≥ 0.5 인 GT 가 2개 이상인 예측 i 가 이 GT 도 Rx ≥ 0.5 로 덮음
  split   (과다분할, over-segmentation) Px(i, j) ≥ 0.5 인 예측이 2개 이상 · 그 조각들의 배타 덮음 합집합이 GT 복셀의 ≥ 50%
  low_iou (TIDE Loc) 최고 IoU_τ ∈ [tide_rules.LOC_MIN_IOU, 매칭 임계] = [0.1, 0.5] (양쪽 포함, 0.1 = TIDE 기본 background_threshold quantify.py:428) — 2D score_2d 와 같은 함수
          tide_rules.is_loc (수정 ⑤ E3, TIDE quantify.py:237 'bg_thresh <= iou <= pos_thresh').
          덮음이 배타적이 되면서 '매칭 안 된 GT 의 최고 IoU < 0.5' 보장은 없어졌다: IoU ≥ 0.5 인 예측이 이웃 GT 에 매칭됐고 이 GT 를
          배타적으로는 덮지 못한 경우(20cm 안 평행 표면 등). TIDE 에서는 그 예측이 TP 이고 이 GT 는 Miss 라 missed 로 둔다.
  missed  (TIDE Miss, main_errors.py:78-81) 그 밖
  경계는 전부 정수 비교(2·count ≥ total) 또는 FEPS 여유.

중복 (dup_of) = TIDE Dupe (수정 ⑤ E2) — tide_rules.pred_error, quantify.py:228-265 순서 그대로
  매칭 안 된 예측 i 에 대해, 무시 아닌 GT(score_frontend: present, quantify.py:18 ex.gt) 중
    Loc  최대 IoU_τ ∈ [LOC_MIN_IOU = 0.1, 0.5] (:236-240 양쪽 포함, Dupe 보다 먼저)
    Dupe 이 keyframe 에서 다른 예측에 매칭된 GT 중 IoU_τ 최대가 ≥ 0.5 (:250-255) → dup_of = 그 GT
    Bkg  최대 IoU_τ < LOC_MIN_IOU (:258-262) · Other 나머지 (:264-265)
  예전(수정 ④): P_τ 가 가장 큰 GT 가 매칭돼 있으면 dup — IoU 를 보지 않아 작은 조각도 중복으로 셌다 (TIDE 가 아님).
  TIDE 는 점수순 탐욕 매칭(quantify.py:74-90)이고 여기는 헝가리안이다. FP 인지 제외인지는 위 FP/제외 규칙이 따로 정한다.

트랙 원인  present keyframe 이 없으면 no_kf. 있으면 그 keyframe 들 상태의 최빈값, 동률이면 STATUS_ORDER(= 판정 순서) 앞의 것.
  옛 값 대응: absorbed → merged, not_detected → missed (OLD_REASON_MAP).
"""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import tide_rules as TR

EPS = 1e-5                                  # 거리 여유 (m) — 모듈 docstring '거리 판정'
FEPS = TR.FEPS                              # 비율·유사도 비교 여유 (TrackEval clear.py:82·86 과 같은 값)
TIE = 1e-9                                  # 최근접 동률 여유 (m)
IOU_THRESHOLD = 0.5
MERGE_MIN_R = 0.5
SPLIT_MIN_P = 0.5
SPLIT_MIN_COVER = 0.5
IGNORE_FRAC = 0.5
STATUS_ORDER = ('merged', 'split', 'low_iou', 'missed')
OLD_REASON_MAP = {'absorbed': 'merged', 'not_detected': 'missed'}


def within(d, tau):
    """d < τ (strict) — float 오차 여유 EPS."""
    return np.asarray(d) < tau - EPS


def tolerant_iou(P, R):
    """IoU_τ = P·R / (P + R − P·R). P+R = 0 이면 0."""
    P = np.asarray(P, np.float64); R = np.asarray(R, np.float64)
    num = P * R
    den = P + R - num
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


class PairStats:
    """keyframe 하나의 (GT j × 예측 i × τ) 복셀 수.
    nP[j, i, t] = 예측 i 복셀 중 GT j 표면에서 τ_t 안인 수 · nR[j, i, t] = GT j 복셀 중 예측 i 에서 τ_t 안인 수
    near[j, i]  = 예측 i 복셀 중 가장 가까운 GT 표면 복셀(주 τ 이내)이 GT j 의 것인 수 · void[i] = 주 τ 안에 GT 표면이 없는 수
    nRx[j, i]   = [E1] GT j 복셀 중 예측 i 가 배타적으로 덮는 수 (주 τ, 모듈 docstring '덮음은 배타적')"""

    def __init__(self, n_pred, n_gt, taus, main):
        self.n_pred, self.n_gt, self.taus, self.main = n_pred, n_gt, tuple(taus), main
        T = len(taus)
        self.nP = np.zeros((len(n_gt), len(n_pred), T), np.int64)
        self.nR = np.zeros((len(n_gt), len(n_pred), T), np.int64)
        self.near = np.zeros((len(n_gt), len(n_pred)), np.int64)
        self.nRx = np.zeros((len(n_gt), len(n_pred)), np.int64)
        self.void = n_pred.copy()
        self._cover = {}                     # (j, i) → GT j 복셀 중 예측 i 에서 주 τ 안인 것의 bool 마스크
        self._xcover = {}                    # (j, i) → GT j 복셀 중 예측 i 가 배타적으로 덮는 것의 bool 마스크

    def ti(self, tau):
        return int(np.argmin(np.abs(np.asarray(self.taus) - tau)))

    def P(self, t):
        return self.nP[:, :, t] / np.maximum(self.n_pred, 1)[None, :]

    def R(self, t):
        return self.nR[:, :, t] / np.maximum(self.n_gt, 1)[:, None]

    def iou(self, t):
        return tolerant_iou(self.P(t), self.R(t))

    def cover(self, j, preds):
        """GT j 복셀 중 preds 의 합집합에서 주 τ 안인 수 (비배타 — 참고용)."""
        return self._union(self._cover, j, preds)

    def xcover(self, j, preds):
        """[E1] GT j 복셀 중 preds 가운데 하나라도 배타적으로 덮는 수 (split 의 조각 합집합)."""
        return self._union(self._xcover, j, preds)

    def _union(self, masks, j, preds):
        m = np.zeros(int(self.n_gt[j]), bool)
        for i in preds:
            c = masks.get((int(j), int(i)))
            if c is not None:
                m |= c
        return int(m.sum())


def pair_stats(pred_vox, gt_vox, taus, main_tau=None):
    """pred_vox: [(N_i,3)] · gt_vox: [(M_j,3)] 셀 중심 → PairStats.
    같은 결과를 내는 가장 단순한 방법은 모든 (i, j) 쌍을 KD-tree 로 조회하는 것이고, 여기서는 결과를 바꾸지 않는 가지치기만 한다:
      ① 모든 GT 표면을 합친 트리로 예측 복셀의 최근접 거리 → τmax 안에 GT 가 하나도 없는 예측 복셀은 어떤 쌍에도 안 들어간다
      ② 모든 예측을 합친 트리로 GT 복셀의 최근접 거리 → 같은 이유로 GT 쪽도 거른다
      ③ 쌍 (i, j) 는 남은 복셀의 bbox 가 τmax 안에서 겹칠 때만, 상대 bbox ± τmax 안의 복셀만 조회한다.
    [E1] 배타 덮음: ① 의 KD-tree 로 GT 복셀마다 모든 예측 복셀까지의 최근접 거리 dall 을 구하고, 주인이 j 인 예측 i 복셀만으로 만든
      트리에서 GT j 복셀의 최근접 거리 dj 를 구해 'dj < 주 τ − EPS 이고 dj ≤ dall + 1e-9' 인 복셀을 센다 (test_match3d M11 이 전체
      거리 행렬로 확인)."""
    taus = tuple(float(t) for t in taus)
    main_tau = taus[len(taus) // 2] if main_tau is None else float(main_tau)
    tmax = max(taus)
    pred_vox = [np.asarray(p, np.float64).reshape(-1, 3) for p in pred_vox]
    gt_vox = [np.asarray(g, np.float64).reshape(-1, 3) for g in gt_vox]
    n_pred = np.array([len(p) for p in pred_vox], np.int64)
    n_gt = np.array([len(g) for g in gt_vox], np.int64)
    st = PairStats(n_pred, n_gt, taus, int(np.argmin(np.abs(np.asarray(taus) - main_tau))))
    tau_arr = np.asarray(taus)
    if not n_pred.sum() or not n_gt.sum():
        return st
    Np, Ng = len(pred_vox), len(gt_vox)
    gcat = np.concatenate(gt_vox)
    pcat = np.concatenate(pred_vox)
    d_pg = cKDTree(gcat).query(pcat, k=1, distance_upper_bound=tmax)[0]
    d_gp, _ = cKDTree(pcat).query(gcat, k=1, distance_upper_bound=tmax)       # GT 복셀 → 모든 예측 복셀 최근접 (E1 의 dall)
    near_p = np.isfinite(d_pg); near_g = np.isfinite(d_gp)
    p_off = np.concatenate(([0], np.cumsum(n_pred))); g_off = np.concatenate(([0], np.cumsum(n_gt)))
    psub = [pred_vox[i][near_p[p_off[i]:p_off[i + 1]]] for i in range(Np)]
    gidx = [np.flatnonzero(near_g[g_off[j]:g_off[j + 1]]) for j in range(Ng)]
    INF = np.full(3, np.inf)
    pb = np.array([[*(s.min(0)), *(s.max(0))] if len(s) else [*INF, *(-INF)] for s in psub]).reshape(Np, 6)
    gb = np.array([[*(gt_vox[j][ix].min(0)), *(gt_vox[j][ix].max(0))] if len(ix) else [*INF, *(-INF)]
                   for j, ix in enumerate(gidx)]).reshape(Ng, 6)
    # bbox 가 τmax 안에서 겹치는 (j, i) 쌍만 (한 번에 벡터 계산)
    ov = np.all(pb[None, :, :3] <= gb[:, None, 3:] + tmax, 2) & np.all(gb[:, None, :3] <= pb[None, :, 3:] + tmax, 2)
    gtree, ptree = {}, {}
    dist_of = [dict() for _ in range(Np)]                          # 예측 i → {GT j: (psub[i] 안 위치, 거리)} — 가장 가까운 GT 라벨용
    for j, i in zip(*np.nonzero(ov)):
        j, i = int(j), int(i)
        lo, hi = gb[j, :3] - tmax, gb[j, 3:] + tmax
        sel = np.flatnonzero(np.all((psub[i] >= lo) & (psub[i] <= hi), 1))
        ps = psub[i][sel]
        if not len(ps):
            continue
        if j not in gtree:
            gtree[j] = cKDTree(gt_vox[j])
        d1 = gtree[j].query(ps, k=1, distance_upper_bound=tmax)[0]
        c1 = (d1[:, None] < tau_arr[None, :] - EPS).sum(0)
        if not c1.any():                                         # τmax 안에 짝이 없으면 R 쪽도 0 (거리는 대칭)
            continue
        st.nP[j, i] = c1
        dist_of[i][j] = (sel, d1)
        lo, hi = pb[i, :3] - tmax, pb[i, 3:] + tmax
        gi = gidx[j][np.all((gt_vox[j][gidx[j]] >= lo) & (gt_vox[j][gidx[j]] <= hi), 1)]
        if not len(gi):
            continue
        if i not in ptree:
            ptree[i] = cKDTree(pred_vox[i])
        d2 = ptree[i].query(gt_vox[j][gi], k=1, distance_upper_bound=tmax)[0]
        w = d2[:, None] < tau_arr[None, :] - EPS
        st.nR[j, i] = w.sum(0)
        if w[:, st.main].any():
            m = np.zeros(int(n_gt[j]), bool); m[gi[w[:, st.main]]] = True
            st._cover[(j, i)] = m
    # 가장 가까운 GT 라벨 (주 τ): 복셀마다 GT 별 최근접 거리의 최소. 거리가 1e-9m 안에서 같으면 GT 번호가 작은 쪽
    # (KD-tree 조회 순서에 따라 달라지지 않게 규칙으로 정한다 — test_match3d M11 이 전체 거리 행렬과 비교)
    owner = [None] * Np                                            # 예측 i → psub[i] 복셀마다 주인 GT 번호 (없으면 -1)
    for i in range(Np):
        if not dist_of[i]:
            continue
        js = sorted(dist_of[i])
        D = np.full((len(psub[i]), len(js)), np.inf)
        for c, j in enumerate(js):
            sel, d1 = dist_of[i][j]
            D[sel, c] = d1
        dmin = D.min(1)
        inm = within(dmin, main_tau)
        if inm.any():
            first = np.argmax(D[inm] <= dmin[inm, None] + TIE, axis=1)    # 최소 거리(±1e-9)를 갖는 첫(가장 작은 번호) GT
            np.add.at(st.near[:, i], np.asarray(js)[first], 1)
            ow = np.full(len(psub[i]), -1, np.int64); ow[inm] = np.asarray(js)[first]
            owner[i] = ow
    st.void = n_pred - st.near.sum(0)
    # [E1] 배타 덮음 — 주인이 j 인 예측 i 복셀이 GT j 복셀의 최근접(모든 예측 복셀 중, 동률 포함)이면 덮음
    main_arr = tau_arr[st.main]
    for j, i in zip(*np.nonzero(st.near)):
        j, i = int(j), int(i)
        own = psub[i][owner[i] == j]
        lo, hi = own.min(0) - main_arr, own.max(0) + main_arr
        gi = gidx[j][np.all((gt_vox[j][gidx[j]] >= lo) & (gt_vox[j][gidx[j]] <= hi), 1)]
        if not len(gi):
            continue
        dj = cKDTree(own).query(gt_vox[j][gi], k=1, distance_upper_bound=tmax)[0]
        ok = within(dj, main_tau) & (dj <= d_gp[g_off[j] + gi] + TIE)
        if ok.any():
            m = np.zeros(int(n_gt[j]), bool); m[gi[ok]] = True
            st._xcover[(j, i)] = m
            st.nRx[j, i] = int(ok.sum())
    return st


def match(sim, threshold=IOU_THRESHOLD):
    """TrackEval clear.py:82-88 형식 1:1 매칭. sim [GT × 예측] → (행 배열, 열 배열)."""
    sim = np.asarray(sim, np.float64)
    if sim.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    score = sim.copy()
    score[score < threshold - FEPS] = 0
    rows, cols = linear_sum_assignment(-score)
    ok = score[rows, cols] > 0 + FEPS
    return rows[ok], cols[ok]


def unmatched_class(n_vox, void, near_col, ignored):
    """매칭 안 된 예측 하나 → ('fp', '') | ('excluded', 이유). near_col[j] = 가장 가까운 표면이 GT j 인 복셀 수, ignored[j] = 무시 GT."""
    n_vox = int(n_vox)
    if n_vox <= 0:
        return 'excluded', 'no_voxels_within_range'
    near_col = np.asarray(near_col, np.int64); ignored = np.asarray(ignored, bool)
    ign_gt = int(near_col[ignored].sum()) if len(near_col) else 0
    void = int(void)
    if 2 * (void + ign_gt) <= n_vox:              # 정확히 절반은 FP (evaluation.py:161 '> 0.5')
        return 'fp', ''
    if 2 * void > n_vox:
        return 'excluded', 'void'
    if 2 * ign_gt > n_vox:
        return 'excluded', 'ignored_gt'
    return 'excluded', 'void+ignored_gt'


def gt_statuses(st, rows, cols, t=None, threshold=IOU_THRESHOLD):
    """매칭 결과 (rows GT, cols 예측) → GT 마다 dict(status, best_iou, best_pred, best_P, best_R, n_frags, split_cover, merged_pred).
    status: tp | merged | split | low_iou | missed (present 여부는 부르는 쪽이 따로 본다)."""
    t = st.main if t is None else t
    Ng, Np = len(st.n_gt), len(st.n_pred)
    iou = st.iou(t) if Np and Ng else np.zeros((Ng, Np))
    P = st.P(t) if Np and Ng else np.zeros((Ng, Np))
    R = st.R(t) if Np and Ng else np.zeros((Ng, Np))
    m_gt = dict(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist()))
    # [E1] merged · split 은 배타 덮음(주 τ) — nRx · near(Px) · xcover. t 는 IoU·P·R(best_*) 에만 쓴다.
    covers = (2 * st.nRx >= st.n_gt[:, None]) & (st.nRx > 0)          # Rx ≥ 0.5 (정수 비교)
    merger = covers.sum(0) >= 2                                       # 예측마다: GT 2개 이상을 각각 Rx ≥ 0.5 로 덮음
    inside = (2 * st.near >= st.n_pred[None, :]) & (st.near > 0)      # Px ≥ 0.5
    out = []
    for j in range(Ng):
        best = int(np.argmax(iou[j])) if Np else -1
        best_iou = float(iou[j, best]) if Np else 0.0
        if best_iou <= 0:
            best = -1
        frags = np.flatnonzero(inside[j]) if Np else np.zeros(0, int)
        cov = 0.0
        if len(frags) >= 2:
            cov = st.xcover(j, frags) / max(int(st.n_gt[j]), 1)
        mp = np.flatnonzero(merger & covers[j]) if Np else np.zeros(0, int)
        if j in m_gt:
            status = 'tp'
        elif len(mp):
            status = 'merged'
        elif len(frags) >= 2 and 2 * st.xcover(j, frags) >= st.n_gt[j]:
            status = 'split'
        elif Np and TR.is_loc(best_iou, threshold):
            status = 'low_iou'
        else:
            status = 'missed'
        out.append(dict(status=status, best_iou=best_iou, best_pred=best,
                        best_P=float(P[j, best]) if best >= 0 else 0.0, best_R=float(R[j, best]) if best >= 0 else 0.0,
                        n_frags=int(len(frags)), split_cover=float(cov), merged_pred=int(mp[0]) if len(mp) else -1,
                        matched_pred=int(m_gt[j]) if j in m_gt else -1))
    return out


def pred_error(st, i, rows, cols, valid=None, t=None, threshold=IOU_THRESHOLD):
    """[E2] 매칭 안 된 예측 i 의 TIDE 오류 → (kind, GT 번호) (tide_rules.pred_error). 매칭된 예측이면 ('', -1).
    valid: GT 마다 무시 아님(present) — None 이면 전부. used = 이 keyframe 매칭(rows)에 쓰인 GT."""
    t = st.main if t is None else t
    if i in np.asarray(cols).tolist():
        return '', -1
    Ng = len(st.n_gt)
    used = np.zeros(Ng, bool); used[np.asarray(rows, int)] = True
    iou = st.iou(t)[:, i] if Ng else np.zeros(0)
    return TR.pred_error(iou, used, valid, pos=threshold)


def duplicate_of(st, i, rows, cols, t=None, valid=None, threshold=IOU_THRESHOLD):
    """[E2] TIDE Dupe: 매칭 안 된 예측 i 가 다른 예측에 매칭된 (무시 아닌) GT 와 IoU_τ ≥ 0.5 면 그 GT 번호, 아니면 -1."""
    kind, j = pred_error(st, i, rows, cols, valid, t, threshold)
    return j if kind == 'dupe' else -1


def track_reason(n_present_kf, statuses):
    """미검출 트랙의 원인. statuses = present keyframe 마다의 상태 (tp 없음)."""
    if not n_present_kf:
        return 'no_kf'
    c = Counter(s for s in statuses if s in STATUS_ORDER)
    if not c:
        return 'missed'
    top = max(c.values())
    return next(s for s in STATUS_ORDER if c.get(s) == top)


def format_values(d):
    """{ep: 실수} → 'ep:값;…' (ep 오름차순, 17 유효숫자 = float64 왕복 정확)."""
    return ';'.join(f'{int(k)}:{float(v):.17g}' for k, v in sorted(d.items()))


def parse_values(s):
    """format_values 의 역. 빈 문자열 → {}."""
    return {int(a): float(b) for a, b in (x.split(':') for x in s.split(';') if x)} if s else {}
