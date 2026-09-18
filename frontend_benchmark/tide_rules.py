#!/usr/bin/env python3
"""TIDE 오류 판정 규칙 한 곳 — 3D(match3d.py)와 2D(score_2d.py)가 같은 상수·같은 비교를 쓴다 (수정 ⑤ E2 · E3).

출처: TIDE (Bolya et al. ECCV 2020, github.com/dbolya/tide) tidecv/quantify.py · tidecv/errors/main_errors.py
      PyPI tidecv 1.0.1 원문 사본 runs/frontend_eval/_audit/fix_E_integrate/followup/refs/ (sdist sha256 ae6a8714…, 줄 번호 기준)
      = _audit/fix_D_match3d/refs/tide_quantify.py · tide_main_errors.py (줄 끝 문자만 다름)

매칭 안 된 예측의 오류 (quantify.py:228-265, 이 순서로 처음 맞는 것 하나)
  :18      ex.gt = 무시(ignore) 아닌 GT 만. 무시 GT 는 오류 판정에 들어가지 않는다
  :230-233 무시 아닌 GT 가 하나도 없으면 Bkg
  :236-240 Loc(BoxError)   idx = IoU 최대 GT, 'self.bg_thresh <= IoU <= self.pos_thresh'   ← 양쪽 경계 포함
  :243-247 Cls             (클래스가 하나라 해당 없음)
  :250-255 Dupe            idx = '이미 매칭에 쓰인(used) GT' 중 IoU 최대, 'IoU >= self.pos_thresh' — 억제한 예측 = 그 GT 의 짝
  :258-262 Bkg             'IoU <= self.bg_thresh' (Loc 을 먼저 보므로 실제로는 IoU < bg)
  :264-265 Other           나머지 (클래스 하나에서 헝가리안 매칭이면 '안 쓴 GT 와 IoU > pos' 뿐 — 최대 가중 매칭에서는 생기지 않는다)
  pos_thresh = 매칭 임계(3D IoU_τ 0.5 · 2D θ 0.5), bg_thresh = LOC_MIN_IOU.
  LOC_MIN_IOU = 0.1 = TIDE 기본값: quantify.py:428 'def __init__(self, pos_threshold:float=0.5, background_threshold:float=0.1, ...)'
  (수정 ⑤ 후속 F2. 그 전에는 score_2d 의 옛 Loc 하한 0.25(θ 최솟값)를 썼다 — 표준 기본값이 아니었다)

매칭 안 된 GT 의 Loc (low_iou) — 3D match3d.gt_statuses · 2D score_2d.score_kf 가 같은 함수 is_loc 을 쓴다
  TIDE 는 GT 쪽 Loc 을 따로 두지 않는다(매칭 안 된 GT 는 다른 오류가 고치지 않으면 Miss, :267-278). 우리 GT 상태의 Loc 은
  '가장 잘 맞는 예측과의 IoU 가 Loc 범위' 라는 뜻으로 :237 의 비교를 그대로 쓴다: bg_thresh ≤ IoU ≤ pos_thresh (양쪽 포함).
  위쪽 경계: 최고 IoU 가 pos 를 넘는데 매칭 안 된 GT 는 그 예측이 이웃 GT 에 쓰인 경우다. TIDE 에서는 그 예측이 TP 라
  오류가 없고 GT 는 Miss 이므로 missed 로 둔다 (3D: 20cm 허용 안의 평행 표면 등, 2D: 픽셀 라벨이 배타적이라 생기지 않음).
  예전: match3d 는 [0.25, 0.5) (포함), score_2d 는 '> min(θ)' (미포함) — 경계값 IoU 정확히 0.25 에서 둘이 달랐다 (감사 fix_D).

비교 여유: FEPS = float64 eps (TrackEval clear.py:82·86 과 같은 값). 경계값에 정확히 걸린 IoU(유리수)가 부동소수 연산으로 조금
  작게 나와도 같은 판정이 나오게. LOC_MIN_IOU 는 함수 안에서 모듈 속성으로 읽는다 (테스트가 한 곳만 바꿔 두 채점기를 확인).
"""
import numpy as np

FEPS = np.finfo('float').eps
LOC_MIN_IOU = 0.1           # bg_thresh — Loc 하한 (포함). 3D · 2D 공통. TIDE 기본 background_threshold (quantify.py:428)


def is_loc(iou, pos, eps=FEPS):
    """LOC_MIN_IOU ≤ IoU ≤ pos (quantify.py:237 'self.bg_thresh <= iou <= self.pos_thresh', 양쪽 포함)."""
    return LOC_MIN_IOU - eps <= float(iou) <= pos + eps


def pred_error(iou, used, valid=None, pos=0.5, eps=FEPS):
    """매칭 안 된 예측 하나의 TIDE 오류.
    iou   (GT 수,) 이 예측과 GT 마다의 IoU · used (GT 수,) bool 그 GT 가 다른 예측에 매칭됨 · valid (GT 수,) bool 무시 아닌 GT (None = 전부)
    → (kind, GT 번호): ('loc', 최대 IoU GT) | ('dupe', 쓰인 GT 중 최대 IoU GT) | ('bkg', -1) | ('other', -1)."""
    iou = np.asarray(iou, np.float64).ravel()
    used = np.asarray(used, bool).ravel()
    valid = np.ones(len(iou), bool) if valid is None else np.asarray(valid, bool).ravel()
    if not valid.any():                                             # :230-233
        return 'bkg', -1
    iv = np.where(valid, iou, -1.0)
    j = int(np.argmax(iv)); mx = float(iv[j])
    if is_loc(mx, pos, eps):                                        # :236-240 (양쪽 포함)
        return 'loc', j
    iu = np.where(valid & used, iou, -1.0)
    ju = int(np.argmax(iu))
    if iu[ju] >= pos - eps:                                         # :250-255
        return 'dupe', ju
    if mx <= LOC_MIN_IOU + eps:                                     # :258-262
        return 'bkg', -1
    return 'other', -1                                              # :264-265
