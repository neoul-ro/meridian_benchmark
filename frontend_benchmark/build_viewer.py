#!/usr/bin/env python3
"""채점 결과를 눈으로 확인하는 뷰어를 만든다 (브라우저로 여는 로컬 폴더, 서버 불필요).

  <RUNS>/viewer/index.html                  ← 시퀀스 목록 (여기서 시작, viewer/index.js 가 목록 데이터)
  <RUNS>/viewer/uHumans2_<seq>/index.html   ← 시퀀스 하나
                              data.js       맵·물체·keyframe 데이터
                              meta.json     목록 페이지·시퀀스 전환이 읽는 작은 기록 (대표 숫자 · 생성 시각 · 출처)
                              kf/<frame>.jpg keyframe 판정 그림 · kf/render_manifest.json (그림마다 입력 내용 키)

사용:  python build_viewer.py [시퀀스...]        뷰어를 만든다 (안 바뀐 그림은 다시 그리지 않는다)
       python build_viewer.py --index-only       뷰어는 그대로, 목록 페이지와 '오래됨' 표시만 1초 안에 갱신

용어는 terms.py 한 곳 (감사 D). 처음 나올 때 한국어(영문).
맵 탭 (위에서 내려다본 2D)
  배경   frontend 가 publish(발행)한 모든 점의 XY 밀도 (5cm 칸)
  물체   GT 물체(같은 gt_object_id 의 GT 트랙(ground-truth track) 묶음, 사람은 GT 트랙 하나씩) 발자국을 3D 판정 색으로:
         검출 · 미검출 · 과소분할 · 채점 제외 (표시 이름은 terms.OBJECT · 색은 terms.COLORS)
         판정 규칙 (object_verdict, 감사 C12 — 색과 이유가 같은 규칙 하나에서 나온다)
           채점 대상(eligible) GT 트랙 = 640×480 crop · 평가 거리 범위 안 가시 픽셀이 한 프레임이라도 area_min_px(1600) 이상
           채점 대상 GT 트랙이 없으면 제외 · 하나라도 검출이면 검출
           아니면 대표 원인 = 채점 대상 GT 트랙들의 미검출 원인 중 가장 많은 값(동률은 terms.REASONS 의 order) → 색 = 그 원인의 group
  필터  판정·종류·미검출 원인으로 고른다. 맞지 않는 물체는 지우지 않고 흐리게 그리고 개수를 함께 보인다
        (사용성 리뷰 M3 — '아예 숨기기' 를 따로 켤 수 있다). 옆 패널에 원인별 '매칭 안 된 물체' 목록과 번호 찾기가 있다
  물체 정보 (object_metrics · object_kf3d, 감사 C13 · C14 · 수정 ⑤)
           3D keyframe 판정 = 그 물체 GT 트랙들의 (keyframe, GT) 상태 개수 (score_kf_gt.csv: TP · 과소분할 · 과다분할 · Loc ·
                      미검출(Miss) · 제외 매칭 · present 아님 — 표시 이름 terms.KF3D)
           accuracy = 채점 대상 GT 트랙에 탐지 인정(detect_credit=1)으로 매칭된 예측 검출(predicted detection)의
                      관측별 평균 거리(dist_mean_cm)의 중앙값 — summary.md 2장 표(관측별 평균의 p50)와 같은 통계
           매칭된 예측 검출 수 = 같은 조건(채점 대상 · 탐지 인정)
           keyframe 중 최대 가시 px (판정 그림이 있는 keyframe 만) 와 전 프레임 최대 가시 px 를 따로 보인다
  궤적   카메라 경로 + keyframe 위치
keyframe 탭
  GT 인스턴스 목록에 2D 판정과 같은 keyframe 의 3D 상태(score_kf_gt.csv)를 함께 보인다.
  이미지 위에 GT 인스턴스 윤곽(2D 판정: TP 실선 · 과다분할/Loc 점선 · 과소분할 · 미검출(Miss))과
  frontend 마스크(TP · 오검출(FP))를 겹친다. void(무시 라벨)에 절반 넘게 떨어진 마스크는 칠하지 않는다.
  판정 근거로 이동 (사용성 리뷰 M4): 물체 정보의 (keyframe, GT) 판정 행(kf_judgment_rows)을 누르면 그 keyframe 으로 가고
  그 물체를 흰 상자로 자동 강조한다. 과소분할은 같은 예측에 함께 묶인 상대 물체를, 과다분할은 조각 예측 번호를 보인다.
  번호 규칙은 한 가지: keyframe 번호(kf)는 score_*.csv 의 kf 열과 같게 0부터, 프레임은 데이터셋 프레임 번호 — 둘 다 이름을 붙인다.
  keyframe 은 2D 상태·3D 상태로 걸러 고를 수 있고, 추적 띠의 칸도 누르면 그 keyframe 으로 간다.
  그림 캐시 (감사 C8): 그림마다 키 = (그 keyframe 의 2D GT·예측 상태 행, 라벨 PNG sha1, frontend_output.h5 sha1, 그리는 코드(render_kf · dashed · 색 · uhumans2.py) digest).
  키가 기록(render_manifest.json)과 다르거나 그림이 없으면 다시 그린다 — 다시 채점해 2D 상태가 바뀌면 그림도 바뀐다.

색: 파랑 #3987e5 · 주황 #d95926 · 청록 #199e70 (terms.COLORS · dataviz 검증: 색약 포함 모든 쌍 구분, 어두운 배경 대비 3:1 이상)
  같은 색이 두 뜻으로 쓰이지 않게 모양으로 나눈다 (사용성 리뷰 m4): GT 윤곽은 선(TP 실선 · 과다분할 긴 점선 · Loc 짧은 점선),
  frontend 마스크는 면(TP 채움 · 무시 GT 매칭 빗금 · 오검출 채움 · void 겹침 무시는 칠하지 않음).

머리말 (사용성 리뷰 m1 · m7): 생성 시각과 채점 시각을 보이고, summary.json 의 채점 단계 키(report_steps)와 다르면
  '채점 결과가 이 뷰어보다 새것' 이라고 경고한다 (file:// 은 viewer/index.js 기준, http 는 summary.json 을 직접 읽어 확인).
  지표 이름·단위·자리수는 terms.METRICS 에서 오고(없는 지표는 뷰어 기본값에 별표), 설명은 눌러서 여는 칸으로 보인다.
URL 해시 (m3): #<시퀀스>/tab=…/obj=…/kf=… — 같은 링크를 열면 같은 화면이 나온다. 다른 시퀀스면 그 폴더로 옮겨 간다.
"""
import argparse
import base64
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import provenance as PV  # noqa: E402
import terms as T  # noqa: E402
from meridian_benchmark.uhumans2 import UH2Sequence  # noqa: E402

CELL = 0.05
AREA_MIN_PX = 1600


def _bgr(h, default):
    """'#3987e5' → OpenCV BGR. 색도 terms.py 한 곳에서 (없으면 지금까지 쓰던 값)."""
    h = str(h or default).lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


COLORS = dict(detected='#3987e5', missed='#d95926', merged='#199e70', ineligible='#898781')
COLORS.update({k: v for k, v in (getattr(T, 'COLORS', None) or {}).items() if k in COLORS})
BLUE, ORANGE, AQUA, GRAY = (_bgr(COLORS['detected'], '#3987e5'), _bgr(COLORS['missed'], '#d95926'),
                            _bgr(COLORS['merged'], '#199e70'), _bgr(COLORS['ineligible'], '#898781'))
CASING = (16, 16, 16)                       # GT 윤곽 아래에 까는 어두운 테두리 (색이 아니라 모양으로 구분, BGR)
HUMAN_COLORS = ('23d5ea',)
MANIFEST = 'render_manifest.json'

# 번호 규칙 한 가지 (수정 ⑦ M4): keyframe 번호 = score_*.csv 의 kf 열 (0부터) · 프레임 = 데이터셋 프레임 번호.
# 둘 다 항상 이름을 붙여 쓴다. 뷰어 템플릿은 이 글자를 data.js 에서 받는다.
KF_LABEL = 'kf'
FRAME_LABEL = '프레임'
GT2D_KEYS = ('tp', 'merged', 'split', 'low_iou', 'missed')
PRED2D_KEYS = ('tp', 'tp_small', 'fp', 'ignore')
KF3D_KEYS = ('tp', 'merged', 'split', 'low_iou', 'missed', 'ignored_match', 'not_present')


def kf_text(k, fr):
    return f'{KF_LABEL} {int(k)} · {FRAME_LABEL} {int(fr)}'


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def xy_cells(p, origin):
    c = np.floor((p[:, :2] - origin) / CELL).astype(np.int32)
    return c


def _px(r):
    return int(float(r.get('max_px_crop') or 0))


# ---------------------------------------------------------------- 물체 판정 (C12 · C13 · C14)
def object_verdict(erows, area_min_px=AREA_MIN_PX):
    """물체 하나의 GT 트랙 행들(score_episodes.csv) → dict(status, rc, reason). 색(status)과 이유(rc·reason)는 같은 규칙 하나에서."""
    if not erows:
        return dict(status='ineligible', rc='', reason='GT 트랙이 실행 구간 밖')
    elig = [r for r in erows if _px(r) >= area_min_px]
    if not elig:
        mx = max(_px(r) for r in erows)
        return dict(status='ineligible', rc='',
                    reason='가시(visible) 픽셀 없음 (640×480 crop · 평가 거리 범위)' if mx == 0 else f'채점 대상 아님 — 최대 {mx}px < {area_min_px}px')
    if any(str(r['detected']) == '1' for r in elig):
        return dict(status='detected', rc='', reason='')
    rc = T.pick_reason([r.get('miss_reason') for r in elig])
    return dict(status=T.reason_group(rc), rc=rc, reason=T.reason_label(rc))


def _credited(o):
    return str(o.get('detect_credit', '1')).strip() not in ('0', '0.0')


def object_metrics(eps, erows, obs_by_ep, best_kf_ep, area_min_px=AREA_MIN_PX):
    """→ dict(n_matched, dist_cm, best_kf, best_kf_px, max_px_all).
    채점 대상 GT 트랙만 본다. 채점 대상이 하나도 없으면(제외 물체) best keyframe · 최대 px 는 전체 GT 트랙에서 (매칭 수는 0)."""
    elig = {int(r['ep_index']) for r in erows if _px(r) >= area_min_px}
    use = elig or {int(r['ep_index']) for r in erows} or set(eps)
    d = [float(o['dist_mean_cm']) for e in eps if e in elig for o in obs_by_ep.get(e, []) if _credited(o)
         and o.get('dist_mean_cm') not in (None, '')]
    bk = max((best_kf_ep[e] for e in eps if e in use and e in best_kf_ep), default=(0, -1))
    return dict(n_matched=len(d), dist_cm=round(float(np.median(d)), 2) if d else None, best_kf=int(bk[1]), best_kf_px=int(bk[0]),
                max_px_all=max((_px(r) for r in erows if int(r['ep_index']) in use), default=0))


def kf3d_counts(kf_gt_rows):
    """score_kf_gt.csv 행 → {ep: {3D 상태: keyframe 수}}."""
    out = defaultdict(dict)
    for r in kf_gt_rows:
        d = out[int(r['ep_index'])]
        d[r['status']] = d.get(r['status'], 0) + 1
    return dict(out)


def object_kf3d(eps, counts):
    """물체의 GT 트랙들 → 3D 상태 개수 합 (0 인 상태는 없음)."""
    tot = {}
    for e in eps:
        for k, v in (counts.get(int(e)) or {}).items():
            tot[k] = tot.get(k, 0) + v
    return tot


# ---------------------------------------------------------------- (keyframe, GT) 판정 행 · 상대 (수정 ⑦ M4)
def _int(v, d=-1):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def kfgt_index(rows):
    """score_kf_gt.csv 행 → (GT 트랙별, keyframe 별) 두 색인."""
    by_ep, by_kf = defaultdict(list), defaultdict(list)
    for r in rows:
        by_ep[_int(r['ep_index'])].append(r)
        by_kf[_int(r['kf'])].append(r)
    return dict(by_ep), dict(by_kf)


def frag_index(obs_rows, eps_of_oid, want=None):
    """score_observations.csv 행 → {(kf, ep): [조각 예측 번호]}.
    조각(fragment) = 복셀 절반 이상(top_frac ≥ 0.5)이 그 GT 물체 소유인 예측 검출 (match3d.gt_statuses 의 inside 와 같은 뜻).
    CSV 는 물체 번호(top_oid)까지만 주므로 같은 물체의 GT 트랙(ep) 에 모두 붙인다. 개수의 정본은 CSV 의 n_frags."""
    out = defaultdict(list)
    for r in obs_rows:
        if float(r.get('top_frac') or 0) < 0.5:
            continue
        oid = _int(r.get('top_oid'))
        k = _int(r.get('kf'))
        for e in eps_of_oid.get(oid, ()):
            if want is None or (k, int(e)) in want:
                out[(k, int(e))].append(_int(r.get('obs')))
    return {k: sorted(v) for k, v in out.items()}


def gt2d_index(gt2_rows):
    """score_2d_gt.csv → {(kf, ep): (2D 상태, 채점 대상 여부, 최고 IoU)}."""
    return {(_int(r['kf']), _int(r['ep_index'])): (r['status'], r['eligible'] == 'True', float(r['best_iou'] or 0))
            for r in gt2_rows}


def kf_judgment_rows(eps, by_ep, by_kf, ep_obj=None, frags=None, st2=None):
    """물체의 GT 트랙들 → (keyframe, GT) 3D 판정 행 [dict(kf, fr, ep, st, obs, px, mates, frags, nfrag)] (kf 오름차순).
      mates  과소분할: 같은 예측 검출에 함께 묶인 다른 물체 번호 (ep_obj 가 있으면 물체 번호, 없으면 ep)
      frags  과다분할: 조각 예측 번호 (frag_index) · nfrag = CSV 의 n_frags"""
    ep_obj = ep_obj or {}
    frags = frags or {}
    out = []
    for e in eps:
        for r in by_ep.get(int(e), ()):
            k, st = _int(r['kf']), r['status']
            s2 = (st2 or {}).get((k, int(e)))
            j = dict(kf=k, fr=_int(r.get('frame')), ep=int(e), st=st, obs=-1, px=_int(r.get('px_label_5m'), 0),
                     mates=[], frags=[], nfrag=0,
                     st2=s2[0] if s2 and s2[1] else '', iou2=round(s2[2], 2) if s2 and s2[1] else -1)
            mo, mt = _int(r.get('merged_obs')), _int(r.get('matched_obs'))
            if st == 'merged' and mo >= 0:
                j['obs'] = mo
                mates = set()
                for o in by_kf.get(k, ()):
                    oe = _int(o['ep_index'])
                    if oe == int(e):
                        continue
                    if mo in (_int(o.get('merged_obs')), _int(o.get('matched_obs'))):
                        mates.add(ep_obj.get(oe, oe) if ep_obj else oe)
                j['mates'] = sorted(x for x in mates if x >= 0)
            elif st == 'split':
                j['nfrag'] = _int(r.get('n_frags'), 0)
                j['frags'] = list(frags.get((k, int(e)), ()))
            elif mt >= 0:
                j['obs'] = mt
            out.append(j)
    return sorted(out, key=lambda j: (j['kf'], j['ep']))


def judgment_counts(js):
    """판정 행 → {상태: 개수} (object_kf3d 와 같아야 한다)."""
    c = {}
    for j in js:
        c[j['st']] = c.get(j['st'], 0) + 1
    return c


def status_counts(rows, keys):
    """행 목록 → {상태: 개수}, 0 인 상태는 빼고 keys 에 없는 값은 그대로 담는다 (조용히 버리지 않음)."""
    c = {}
    for r in rows:
        st = r['status'] if isinstance(r, dict) else r
        c[st] = c.get(st, 0) + 1
    return {k: c[k] for k in list(keys) + sorted(set(c) - set(keys)) if c.get(k)}


def kf_boxes(lab):
    """라벨 그림 한 장 → {ep: [x0, y0, x1, y1]} (라벨 값 = ep + 1). 2D 채점 대상이 아닌 GT 도 포함 — 이동 후 강조용."""
    if lab is None:
        return {}
    ys, xs = np.nonzero(lab)
    if not len(ys):
        return {}
    v = np.asarray(lab)[ys, xs].astype(np.int64) - 1
    out = {}
    order = np.argsort(v, kind='stable')
    v, xs, ys = v[order], xs[order], ys[order]
    edge = np.flatnonzero(np.diff(v)) + 1
    for a, b in zip(np.r_[0, edge], np.r_[edge, len(v)]):
        out[int(v[a])] = [int(xs[a:b].min()), int(ys[a:b].min()), int(xs[a:b].max()) + 1, int(ys[a:b].max()) + 1]
    return out


# ---------------------------------------------------------------- 생성 시각 · 오래됨 (수정 ⑦ m1)
def build_stamp(summary=None, summary_path=None):
    """data.js 에 넣을 출처 기록. steps = 그 시퀀스 채점 단계의 캐시 키 (summary.json report_steps)."""
    steps = {k: v.get('key') for k, v in ((summary or {}).get('report_steps') or {}).items() if isinstance(v, dict)}
    fin = [v.get('finished') for v in ((summary or {}).get('report_steps') or {}).values() if isinstance(v, dict) and v.get('finished')]
    mt = 0.0
    try:
        mt = Path(summary_path).stat().st_mtime if summary_path else 0.0
    except OSError:
        mt = 0.0
    return dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), generated_epoch=float(time.time()),
                steps=steps, scored=max(fin) if fin else '', summary_mtime=round(mt, 3))


def staleness(built, summary):
    """뷰어(built)가 지금 채점 결과(summary 의 그 시퀀스 항목)보다 오래됐는지.
    → dict(state='fresh'|'stale'|'unknown', steps=[다른 단계], detail)"""
    now = {k: v.get('key') for k, v in ((summary or {}).get('report_steps') or {}).items() if isinstance(v, dict)}
    old = (built or {}).get('steps') or {}
    if not summary or not now:
        return dict(state='unknown', steps=[], detail='summary.json 에 이 시퀀스 채점 기록이 없습니다')
    diff = sorted(k for k in set(now) | set(old) if now.get(k) != old.get(k))
    if diff:
        return dict(state='stale', steps=diff, detail='채점이 다시 돌았습니다 (' + ', '.join(diff) + ')')
    return dict(state='fresh', steps=[], detail='채점 결과와 같은 입력으로 만든 뷰어입니다')


# ---------------------------------------------------------------- 머리말 지표 (표시 이름은 terms.py, 없으면 눈에 띄게 대체)
def pct(v, nd=1):
    return '–' if v is None else f'{v * 100:.{nd}f}%'


def pct_kn(kn, nd=1):
    """개수에서 한 번만 반올림 (half-up, 정수 연산) — summary.md · 뷰어가 같은 규칙 (감사 C15)."""
    if not kn or not kn[1]:
        return '–'
    k, n = int(kn[0]), int(kn[1])
    scale = 10 ** nd
    return f'{(2 * 100 * scale * k + n) // (2 * n) / scale:.{nd}f}%'


def num(v, nd=3):
    return '–' if v is None else f'{float(v):.{nd}f}'


def same_sentence(a, b):
    """두 설명이 사실상 같은 문장인지 (N7: 머리말 설명 칸에 같은 말이 두 번 나오지 않게)."""
    import difflib
    import re as _re
    na, nb = (_re.sub(r'[^0-9A-Za-z가-힣]', '', s or '') for s in (a, b))
    if not na or not nb:
        return False
    return na in nb or nb in na or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.7


def join_desc(gloss, desc):
    """terms 의 쉬운 뜻 + 뷰어의 자세한 설명. 두 글이 같은 말이면 한 번만 쓴다 (N7).
    아래 desc 는 쉬운 뜻을 되풀이하지 않고 모집단·기준·분모만 적는다."""
    if not gloss:
        return desc
    if not desc or same_sentence(gloss, desc):
        return gloss
    return f'{gloss} · {desc}'


def term_key(keys):
    """terms.METRICS 에 있는 첫 이름 (⑥ 가 지표를 더하면 뷰어가 바로 그 이름·자리수를 쓴다). 없으면 ''."""
    tbl = getattr(T, 'METRICS', None) or {}
    for k in (keys if isinstance(keys, (tuple, list)) else (keys,)):
        m = tbl.get(k)
        if isinstance(m, dict) and m.get('label'):
            return k
    return ''


def term_metric(keys):
    """terms.py METRICS 항목 → dict(label 짧은 이름, full 모집단까지, gloss, kind). 없으면 None (뷰어 기본값 + 별표)."""
    key = term_key(keys)
    if not key:
        return None
    m = T.METRICS[key]
    full = T.metric_label(key) if callable(getattr(T, 'metric_label', None)) else m['label']
    short = m['label']
    gl = T.metric_gloss(key) if callable(getattr(T, 'metric_gloss', None)) else m.get('gloss', '')
    return dict(label=short, full=full, gloss=gl, kind=m.get('kind', ''))


def term_fmt(keys, v, kind, kn=None):
    """표시 글자 — terms.py 에 지표가 있으면 그 자리수 규칙으로 (summary.md 와 같은 글자), 없으면 뷰어 기본 규칙."""
    key = term_key(keys)
    if kn is not None:
        return (T.fmt_kn(key, kn[0], kn[1]) if key and callable(getattr(T, 'fmt_kn', None)) else pct_kn(kn)) if kn and kn[1] else '–'
    if key and callable(getattr(T, 'fmt', None)):
        return T.fmt(key, v)
    if callable(getattr(T, 'fmt_by_kind', None)):
        return T.fmt_by_kind(kind, v)
    return pct(v) if kind == 'percent' else num(v, 2) if kind == 'score' else ('–' if v is None else str(v))


def _kind_label():
    """'정적 / 사람' 칸 이름 — terms.METRICS 의 모집단 글자를 쓴다 (둘 중 하나라도 없으면 뷰어 기본값)."""
    a = (getattr(T, 'METRICS', None) or {}).get('track_recall_static') or {}
    b = (getattr(T, 'METRICS', None) or {}).get('track_recall_human') or {}
    pa, pb = str(a.get('population', '')).split('·')[-1].strip(), str(b.get('population', '')).split('·')[-1].strip()
    return f'{pa} / {pb}' if pa and pb else '정적 / 사람'


def cmp_note(s3, s2):
    """3D 와 2D 판정이 다를 수 있는 이유 한 줄 (docs/METRICS.md '3D 매칭 기준' · '2D 채점' 과 같은 근거, 값은 params 에서)."""
    p3, p2 = s3.get('params') or {}, s2.get('params') or {}
    tau = int(round(float(p3.get('tau_m', 0.2)) * 100))
    vox = float(p3.get('voxel_m', 0.02)) * 100
    rng = float(p3.get('max_range_m', 5.0))
    thr = p3.get('iou_threshold', 0.5)
    th2 = p2.get('theta', 0.5)
    return (f'3D 와 2D 는 비교하는 것이 다릅니다 — 3D 는 그 keyframe 라벨을 깊이로 역투영한 GT 표면을 {vox:g}cm 복셀로 만들고, '
            f'허용 거리 τ={tau}cm 의 IoU_τ ≥ {thr} 로 맞춥니다 ({rng:g}m 안). '
            f'2D 는 같은 keyframe 의 마스크 픽셀을 IoU > {th2} (void 칸은 예측에서 뺌) 로 맞춥니다. '
            f'허용 거리와 기준이 달라 얇거나 깊이가 튀는 물체는 한쪽에서만 매칭될 수 있습니다 (docs/METRICS.md).')


def metric_rows(S, ctx):
    """머리말 지표 [dict(key, label, value, desc, fallback)]. 이름·자리수는 terms.py, 없으면 뷰어 기본값(fallback=True)."""
    area, tau, ftau, theta, iou3 = ctx['area_min_px'], ctx['tau_cm'], ctx['f_tau_cm'], ctx['theta'], ctx['iou_thr']
    kn3 = S.get('recall_3d_kn') or [None, None]
    base = [
        ('recall_3d', ('track_recall_eligible', 'track_recall'), '트랙 재현율 (전체 채점 대상)',
         term_fmt(('track_recall_eligible', 'track_recall'), S.get('recall_3d'), 'percent', S.get('recall_3d_kn')),
         f'모집단 = 채점 대상(eligible) GT 트랙(ground-truth track, 640×480 crop 에 {area}px 이상 보인 적 있음) · '
         f'있음(present) = 그 keyframe 라벨 5m 안 {area}px 이상 · 매칭 = GT 표면과 허용 거리 IoU_τ ≥ {iou3} (τ {tau}cm) · '
         f'분자/분모 {kn3[0]}/{kn3[1]}'),
        ('recall_3d_kind', ('track_recall_static',), _kind_label(),
         term_fmt(('track_recall_static',), S.get('recall_3d_static'), 'percent', S.get('recall_3d_static_kn')) + ' / ' +
         term_fmt(('track_recall_human',), S.get('recall_3d_human'), 'percent', S.get('recall_3d_human_kn')),
         '왼쪽이 정적 물체, 오른쪽이 사람입니다 · 사람이 없는 시퀀스는 –'),
        ('within_tau', ('within_tau', 'p_tau', 'within_tau_fraction'), f'P@{tau}cm',
         term_fmt(('within_tau', 'p_tau', 'within_tau_fraction'), S.get('within20'), 'percent'),
         f'분모는 매칭된 예측 검출의 복셀 수입니다 (τ {tau}cm)'),
        ('f_tau', ('f20', 'f_tau'), f'F@{ftau}cm', term_fmt(('f20', 'f_tau'), S.get('f20'), 'score'),
         f'같은 keyframe GT 표면 기준 P@{ftau}cm·R@{ftau}cm 의 조화평균 (Tanks and Temples), 관측별 중앙값'),
        ('recall_2d', ('recall_2d', 'seg_recall_2d'), f'2D 재현율 @IoU{theta}',
         term_fmt(('recall_2d', 'seg_recall_2d'), S.get('recall_2d'), 'percent'),
         f'매칭 = 마스크 IoU > {theta} (void 칸은 예측에서 뺌) · 분모 {S.get("n_gt_2d", "–")}개'),
        ('pq', ('pq',), '2D PQ', term_fmt(('pq',), S.get('pq'), 'score'), 'PQ(Panoptic Quality) = SQ × RQ (Kirillov et al. 2019)'),
        ('idf1', ('idf1',), '추적 IDF1', term_fmt(('idf1',), S.get('idf1'), 'score'),
         'IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN), keyframe 단위 MOT(multi-object tracking, 다중 물체 추적) (Ristani et al. 2016)'),
        ('hota', ('hota_alpha', 'hota'), 'HOTA_α', term_fmt(('hota_alpha', 'hota'), S.get('hota'), 'score'),
         'α 한 점(유사도 임계)에서의 √(DetA_α·AssA_α). 논문의 최종 HOTA(α 19점 평균)가 아님'),
        ('n_kf', ('n_kf', 'keyframes'), 'keyframe', str(S.get('n_kf', '–')),
         f'{KF_LABEL} 번호는 score_*.csv 의 kf 열과 같게 0부터 셉니다 (데이터셋 {FRAME_LABEL} 번호와 다릅니다)'),
    ]
    rows = []
    for key, tkey, label, value, desc in base:
        t = term_metric(tkey)
        if key == 'recall_3d_kind':                                     # 이름은 두 지표의 모집단 글자에서 만든다
            t = dict(label=label, full=label, gloss=(t or {}).get('gloss', '')) \
                if term_key(('track_recall_static',)) and term_key(('track_recall_human',)) else None
        rows.append(dict(key=key, label=(t or {}).get('label') or label, full=(t or {}).get('full') or label, value=value,
                         desc=join_desc((t or {}).get('gloss') or '', desc), fallback=t is None))
    return rows


# ---------------------------------------------------------------- 시퀀스 목록 페이지 (수정 ⑦ m2)
INDEX_TEMPLATE = 'viewer_index_template.html'


def viewer_meta(data):
    """data.js → 목록 페이지·시퀀스 전환이 쓰는 작은 기록 (meta.json). 큰 배열은 넣지 않는다."""
    return dict(seq=data['seq'], name=data.get('seq_name') or data['seq'], dir=f'uHumans2_{data["seq"]}', generated=data['generated'],
                generated_epoch=data['generated_epoch'], scored=data['source'].get('scored', ''),
                stats=[dict(key=r['key'], label=r['label'], full=r['full'], value=r['value']) for r in data['metrics']],
                n_obj=len(data['objects']), n_kf=len(data['kfs']), source=data['source'])


def read_summary(summary_path):
    """summary.json → {시퀀스: 항목}. 없거나 깨졌으면 {} (뷰어는 계속 만든다)."""
    try:
        d = json.loads(Path(summary_path).read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in d.items() if isinstance(v, dict) and 'report_steps' in v} if isinstance(d, dict) else {}


def write_index(viewer_dir, summary_path):
    """viewer/index.html (시퀀스 목록) + viewer/index.js (목록 데이터 · 시퀀스 전환 · 오래됨 표시) 를 쓴다.
    file:// 에서 서버 없이 열려야 하므로 fetch 를 쓰지 않고 <script> 로 읽는다."""
    viewer_dir = Path(viewer_dir)
    summ = read_summary(summary_path)
    seqs = []
    for md in sorted(viewer_dir.glob('uHumans2_*/meta.json')):
        try:
            m = json.loads(md.read_text())
        except (OSError, ValueError):
            continue
        m['link'] = f'{md.parent.name}/index.html'
        m['stale'] = staleness(m.get('source') or {}, summ.get(m.get('seq')))
        seqs.append(m)
    seqs.sort(key=lambda m: m['seq'])
    try:                                                           # 사람에게는 뷰어 폴더 기준 상대 경로로 (내부 절대 경로를 보이지 않는다)
        rel = os.path.relpath(Path(summary_path), viewer_dir)
    except ValueError:
        rel = Path(summary_path).name
    payload = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), generated_epoch=float(time.time()),
                   summary=rel, summary_exists=bool(summ),
                   summary_mtime=round(Path(summary_path).stat().st_mtime, 3) if Path(summary_path).exists() else 0.0,
                   kf_label=KF_LABEL, frame_label=FRAME_LABEL, seqs=seqs)
    (viewer_dir / 'index.js').write_text('window.FB_INDEX=' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n')
    shutil.copy(P.HERE / INDEX_TEMPLATE, viewer_dir / 'index.html')
    return viewer_dir / 'index.html'


# ---------------------------------------------------------------- keyframe 그림 캐시 (C8)
def kf_render_key(fi, gt_rows, pred_rows, label_sha1, h5_sha1, code_sha1):
    g = sorted((str(r['ep_index']), str(r['status']), str(r['eligible'])) for r in gt_rows)
    p = sorted((str(r['pred']), str(r['status'])) for r in pred_rows)
    return PV.key_of([int(fi), g, p, label_sha1, h5_sha1, code_sha1])


def load_render_manifest(kf_dir):
    try:
        return json.loads((Path(kf_dir) / MANIFEST).read_text())
    except (OSError, ValueError):
        return {}


def save_render_manifest(kf_dir, keys):
    return PV.write_json_if_changed(Path(kf_dir) / MANIFEST, {str(int(k)): v for k, v in sorted(keys.items())}, indent=0)


def stale_kf_frames(kf_dir, keys):
    """keys {frame: 키} 중 다시 그려야 할 frame (기록 없음 · 키 다름 · 그림 없음)."""
    man = load_render_manifest(kf_dir)
    return sorted(int(fr) for fr, k in keys.items() if man.get(str(int(fr))) != k or not (Path(kf_dir) / f'{int(fr):06d}.jpg').exists())


# ---------------------------------------------------------------- keyframe 그림
W = {}


def _init(seq_dir, run_h5, labels_dir, out_dir):
    W['seq'] = UH2Sequence(seq_dir)
    W['f'] = h5py.File(run_h5, 'r')
    W['labels'] = Path(labels_dir); W['out'] = Path(out_dir)


def dashed(img, cnt, color, on=6, off=5, thick=2):
    pts = cnt.reshape(-1, 2)
    n = len(pts)
    i = 0
    while i < n:
        seg = pts[i:min(i + on, n)]
        if len(seg) > 1:
            cv2.polylines(img, [seg.reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)
        i += on + off


def render_kf(job):
    k, fi, o0, no, gt_rows, pred_rows = job
    seq, f = W['seq'], W['f']
    rgb = seq.rgb(fi)[:, 40:680]
    img = cv2.cvtColor((cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) * 0.5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    masks = np.unpackbits(f['obs/mask_bits'][o0:o0 + no], axis=1)[:, :192 * 256].reshape(-1, 192, 256)
    over = img.copy()
    yy, xx = np.mgrid[0:480, 0:640]
    stripe = ((xx + yy) % 8) < 4                             # 제외 매칭은 빗금 (수정 ⑦ m4 — 같은 파랑이 두 뜻으로 쓰이지 않게)
    for pr in pred_rows:
        st = pr['status']
        if st in ('ignore',):
            continue
        m = cv2.resize(masks[int(pr['pred'])], (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
        if st == 'tp_small':
            over[m & stripe] = BLUE
        else:
            over[m] = BLUE if st == 'tp' else ORANGE
    img = cv2.addWeighted(over, 0.38, img, 0.62, 0)
    lab = cv2.imread(str(W['labels'] / f'{fi:06d}.png'), cv2.IMREAD_UNCHANGED)
    for g in gt_rows:
        m = (lab == int(g['ep_index']) + 1).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        st = g['status']
        # GT 윤곽은 어두운 테두리(casing) 위에 그린다 — 같은 파랑 마스크 위에 겹쳐도 선이 보인다 (사용성 2차 리뷰 5)
        if g['eligible'] != 'True':
            cv2.drawContours(img, cnts, -1, CASING, 3, cv2.LINE_AA)
            cv2.drawContours(img, cnts, -1, GRAY, 1, cv2.LINE_AA)
        elif st == 'tp':
            cv2.drawContours(img, cnts, -1, CASING, 4, cv2.LINE_AA)
            cv2.drawContours(img, cnts, -1, BLUE, 2, cv2.LINE_AA)
        elif st == 'split':                                  # 긴 점선 (수정 ⑦ m4 — Loc 와 선 모양으로 구분)
            for c in cnts:
                dashed(img, c, CASING, on=16, off=7, thick=4)
                dashed(img, c, BLUE, on=16, off=7)
        elif st == 'low_iou':                                # 짧은 점선
            for c in cnts:
                dashed(img, c, CASING, on=4, off=6, thick=4)
                dashed(img, c, BLUE, on=4, off=6)
        elif st == 'merged':
            cv2.drawContours(img, cnts, -1, CASING, 4, cv2.LINE_AA)
            cv2.drawContours(img, cnts, -1, AQUA, 2, cv2.LINE_AA)
        else:
            cv2.drawContours(img, cnts, -1, CASING, 4, cv2.LINE_AA)
            cv2.drawContours(img, cnts, -1, ORANGE, 2, cv2.LINE_AA)
    cv2.imwrite(str(W['out'] / f'{fi:06d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return k


# ---------------------------------------------------------------- 데이터
def build(s, workers=8, log=print):
    t0 = time.time()
    run, gt_path, labels = P.run_dir(s), P.gt_h5(s), P.RUNS / 'gt_labels_2d' / f'uHumans2_{s}'
    out = P.RUNS / 'viewer' / f'uHumans2_{s}'
    (out / 'kf').mkdir(parents=True, exist_ok=True)
    H = PV.Hasher(P.RUNS / '.cache' / 'sha1.json')
    seq = UH2Sequence(P.seq_dir(s))
    f = h5py.File(run / 'frontend_output.h5', 'r')
    g = h5py.File(gt_path, 'r'); ix = g['index']
    ep_tid = ix['tracklet_id'][:]; ep_oid = ix['gt_object_id'][:]
    ep_color = [str((lambda c: c.decode() if isinstance(c, bytes) else c)(
        g[f'tracklets/{int(t):05d}/_metadata/color_hex'][()])).lstrip('#') for t in ep_tid]
    s3 = json.loads((run / 'score.json').read_text())
    s2 = json.loads((run / 'score_2d.json').read_text())
    area = int((s3.get('visibility') or {}).get('area_min_px') or AREA_MIN_PX)
    tau_m = float((s3.get('params') or {}).get('tau_m', 0.2))
    match_key = f'match@{int(round(tau_m * 100))}'
    ep3 = {int(r['ep_index']): r for r in rows(run / 'score_episodes.csv')}
    obs3 = rows(run / 'score_observations.csv')
    gt2 = rows(run / 'score_2d_gt.csv'); pr2 = rows(run / 'score_2d_pred.csv')
    mot = json.loads((run / 'score_mot.json').read_text()) if (run / 'score_mot.json').exists() else None
    tl = json.loads((run / 'score_mot_timeline.json').read_text()).get('episode', {}) if (run / 'score_mot_timeline.json').exists() else {}
    geo = json.loads((run / 'score_geometry.json').read_text()) if (run / 'score_geometry.json').exists() else None
    kfgt = rows(run / 'score_kf_gt.csv') if (run / 'score_kf_gt.csv').exists() and (run / 'score_kf_gt.csv').stat().st_size else []
    kf3 = kf3d_counts(kfgt)
    s3_of = {(int(r['kf']), int(r['ep_index'])): r['status'] for r in kfgt}

    kf_frame = f['kf/frame_idx'][:]; o0 = f['kf/obs_start'][:]; no = f['kf/n_obs'][:]
    pn = f['obs/points_num'][:]

    # 좌표 원점: 예측 점 전체의 최소 XY
    pts = f['points'][:]
    step = max(1, len(pts) // 4_000_000)
    sub = pts[::step].astype(np.float64)
    origin = np.floor(sub[:, :2].min(0) / CELL) * CELL - 1.0

    # 배경 = publish 된 모든 점의 XY 밀도
    c = xy_cells(sub, origin)
    keys, cnt = np.unique(c[:, 0].astype(np.int64) << 16 | c[:, 1], return_counts=True)
    bg = np.stack([(keys >> 16), keys & 0xFFFF], 1).astype(np.int16)

    # 예측 점: 3D 매칭된 예측 검출 / 매칭 안 됨
    matched_obs = np.zeros(len(pn), bool)
    for r in obs3:
        if r.get(match_key) not in (None, '', '0'):
            matched_obs[int(r['obs'])] = True
    owner = np.repeat(np.arange(len(pn)), pn)[::step]
    pred_layers = {}
    for name, sel in (('matched', matched_obs[owner]), ('unmatched', ~matched_obs[owner])):
        cc = c[sel]
        kk = np.unique(cc[:, 0].astype(np.int64) << 16 | cc[:, 1])
        pred_layers[name] = np.stack([kk >> 16, kk & 0xFFFF], 1).astype(np.int16)

    # 물체: 정적 물체는 gt_object_id 단위, 사람은 GT 트랙 단위
    groups = defaultdict(list)
    for e in range(len(ep_tid)):
        human = ep_color[e] in HUMAN_COLORS
        groups[('h', e) if human else ('s', int(ep_oid[e]))].append(e)
    ep_to_obj = {}
    for idx, eps_ in enumerate(groups.values()):
        for e in eps_:
            ep_to_obj[e] = idx
    # (keyframe, GT) 판정 행과 상대 (M4)
    kg_by_ep, kg_by_kf = kfgt_index(kfgt)
    eps_of_oid = defaultdict(list)
    for e in range(len(ep_tid)):
        eps_of_oid[int(ep_oid[e])].append(e)
    split_pairs = {(int(r['kf']), int(r['ep_index'])) for r in kfgt if r['status'] == 'split'}
    frags = frag_index(obs3, eps_of_oid, split_pairs) if split_pairs else {}
    st2_of = gt2d_index(gt2)                                   # 같은 (kf, GT) 의 2D 상태 (2D·3D 가 다를 때 함께 보여 준다)
    vis = np.load(P.vis_npz(s))
    kf_of_frame = {int(fr): k for k, fr in enumerate(kf_frame)}
    best_kf_ep = {}
    for e_, fr, pc in zip(vis['ep_index'], vis['frame'], vis['px_crop']):
        if int(fr) in kf_of_frame and pc > best_kf_ep.get(int(e_), (0, -1))[0]:
            best_kf_ep[int(e_)] = (int(pc), kf_of_frame[int(fr)])
    obs_by_ep = defaultdict(list)
    for r in obs3:
        if r.get('ep_index') not in (None, '') and r.get(match_key) not in (None, '', '0'):
            obs_by_ep[int(r['ep_index'])].append(r)

    objects, cells_all = [], []
    off = 0
    reasons_seen = {}
    for (kind, key), eps in groups.items():
        p = np.concatenate([g[f'tracklets/{int(ep_tid[e]):05d}/tracklet_geometry/points'][:] for e in eps]).astype(np.float64)
        oc = xy_cells(p, origin)
        kk = np.unique(oc[:, 0].astype(np.int64) << 16 | oc[:, 1])
        cells = np.stack([kk >> 16, kk & 0xFFFF], 1).astype(np.int16)
        erows = [ep3[e] for e in eps if e in ep3]
        v = object_verdict(erows, area)
        mtr = object_metrics(eps, erows, obs_by_ep, best_kf_ep, area)
        if v['rc']:
            reasons_seen[v['rc']] = T.reason_label(v['rc'])
        tracks = []
        for e in eps:
            seq_ = tl.get(str(e))
            if not seq_:
                continue
            ids = [p_ for _, p_ in seq_]
            m_ids = [x for x in ids if x is not None]
            sw = sum(1 for a_, b_ in zip(m_ids, m_ids[1:]) if a_ != b_)
            tracks.append(dict(first=int(ep3[e]['first_frame']) if e in ep3 else -1, kfs=[int(k_) for k_, _ in seq_],
                               ids=[-1 if p_ is None else int(p_) for _, p_ in seq_], switches=sw))

        def ep_status(r):
            if _px(r) < area:
                return '제외(ignored)'
            return '검출' if str(r['detected']) == '1' else T.reason_label(r.get('miss_reason'))
        ctr = p[:, :2].mean(0)
        kfj = [{k_: v_ for k_, v_ in j.items()                          # kf · 프레임 · ep · 상태는 0 이어도 남긴다
                if k_ in ('kf', 'fr', 'ep', 'st') or v_ not in (0, -1, [], '')}
               for j in kf_judgment_rows(eps, kg_by_ep, kg_by_kf, ep_to_obj, frags, st2_of)]
        objects.append(dict(
            id=int(ep_oid[eps[0]]), kind='human' if kind == 'h' else 'static', status=v['status'], reason=v['reason'],
            off=off, n=len(cells), cx=round(float((ctr[0] - origin[0]) / CELL), 1), cy=round(float((ctr[1] - origin[1]) / CELL), 1),
            size=[round(float(x), 2) for x in (p.max(0) - p.min(0))], best_kf=mtr['best_kf'], best_px=mtr['best_kf_px'],
            max_px=mtr['max_px_all'], n_matched=mtr['n_matched'], dist_cm=mtr['dist_cm'], rc=v['rc'], tracks=tracks,
            kf3d=object_kf3d(eps, kf3), kfj=kfj, eplist=[int(e) for e in eps],
            eps=[dict(ep=int(r['ep_index']), first=int(r['first_frame']), last=int(r['last_frame']), status=ep_status(r), px=_px(r))
                 for r in erows]))
        cells_all.append(cells); off += len(cells)
    obj_cells = np.concatenate(cells_all) if cells_all else np.zeros((0, 2), np.int16)

    # 판정 행이 가리키는 예측 마스크의 상자 (M4 나머지: 조각이 어느 마스크인지 이미지에서 보이게)
    need = defaultdict(set)
    for o in objects:
        for j in o['kfj']:
            if j.get('frags'):
                need[j['kf']].update(int(x) for x in j['frags'])
            if j.get('st') == 'merged' and int(j.get('obs', -1)) >= 0:
                need[j['kf']].add(int(j['obs']))
    pboxes = defaultdict(dict)
    for k, obs_set in need.items():
        n_k = int(no[k])
        if not n_k:
            continue
        bits = f['obs/mask_bits'][int(o0[k]):int(o0[k]) + n_k]
        masks = np.unpackbits(bits, axis=1)[:, :192 * 256].reshape(-1, 192, 256)
        for ob in sorted(obs_set):
            li = int(ob) - int(o0[k])                          # 전역 관측 번호 = 그 keyframe 의 obs_start + 마스크 순번
            if not (0 <= li < len(masks)):
                continue
            ys, xs = np.nonzero(masks[li])
            if not len(ys):
                continue
            pboxes[k][str(int(ob))] = [int(xs.min() * 2.5), int(ys.min() * 2.5),
                                       int((xs.max() + 1) * 2.5), int((ys.max() + 1) * 2.5)]   # 192×256 → 480×640

    # 궤적 · keyframe
    fr_all = np.arange(0, seq.n_frames, 5)
    tw, _ = seq.camera_poses(seq.stamps_ns[fr_all], cam='left_cam')
    traj = np.round((tw[:, :2] - origin) / CELL * 4).astype(np.int16)
    tk, qk = seq.camera_poses(seq.stamps_ns[kf_frame], cam='left_cam')
    from scipy.spatial.transform import Rotation
    fwd = Rotation.from_quat(qk).apply([0, 0, 1])                       # optical z = 카메라 정면
    gt_by_kf = defaultdict(list); pr_by_kf = defaultdict(list)
    for r in gt2:
        gt_by_kf[int(r['kf'])].append(r)
    for r in pr2:
        pr_by_kf[int(r['kf'])].append(r)
    kfs = []
    for k, fr in enumerate(kf_frame):
        gs = [r for r in gt_by_kf[k] if r['eligible'] == 'True']
        lab = cv2.imread(str(labels / f'{int(fr):06d}.png'), cv2.IMREAD_UNCHANGED)
        boxes = kf_boxes(lab)            # 2D 채점 대상이 아닌 GT 도 (판정 근거 keyframe 으로 이동한 뒤 강조하려면 필요, M4)
        kfs.append(dict(
            frame=int(fr), x=round(float((tk[k, 0] - origin[0]) / CELL), 1), y=round(float((tk[k, 1] - origin[1]) / CELL), 1),
            yaw=round(float(np.arctan2(fwd[k, 1], fwd[k, 0])), 3), n_obs=int(no[k]),
            gt=status_counts(gs, GT2D_KEYS),                                  # 2D 채점 대상 GT 상태 (과다분할·Loc 를 합치지 않는다)
            s3=status_counts(kg_by_kf.get(k, ()), KF3D_KEYS),                 # 같은 keyframe 의 3D 상태
            pred=status_counts(pr_by_kf[k], PRED2D_KEYS),
            boxes={str(e): b for e, b in boxes.items() if e in ep_to_obj}, pboxes=dict(pboxes.get(k, {})),
            inst=[dict(obj=ep_to_obj.get(int(r['ep_index']), -1), ep=int(r['ep_index']), st=r['status'], iou=float(r['best_iou']),
                       s3=s3_of.get((k, int(r['ep_index'])), ''), el=1 if r['eligible'] == 'True' else 0,
                       px=int(float(r['area_cells']) * 6.25),
                       bp=int(float(r['best_pred'])) if r.get('best_pred') not in (None, '') else -1,
                       mp=int(float(r['matched_pred'])) if r.get('matched_pred') not in (None, '') else -1,
                       lv=int(float(r['level'])) if r.get('level') not in (None, '') else -1,
                       tr=float(r['trunc']) if r.get('trunc') not in (None, '') else -1,
                       oc=float(r['occ']) if r.get('occ') not in (None, '') else -1,
                       md=int(float(r['min_dim'])) if r.get('min_dim') not in (None, '') else -1) for r in gt_by_kf[k]]))

    # keyframe 그림 — 입력 내용 키가 바뀐 것만 (C8)
    h5_sha = H.file(run / 'frontend_output.h5')
    import inspect
    import meridian_benchmark.uhumans2 as U2
    code_sha = PV.sha1_text(inspect.getsource(render_kf) + inspect.getsource(dashed) + repr((BLUE, ORANGE, AQUA, GRAY, CASING))
                            + PV.ast_digest(U2.__file__))                # 그림을 그리는 코드만 (build 의 다른 곳을 고쳐도 다시 안 그림)
    rkeys = {}
    for k, fr in enumerate(kf_frame):
        lp = labels / f'{int(fr):06d}.png'
        rkeys[int(fr)] = kf_render_key(int(fr), gt_by_kf[k], pr_by_kf[k], H.file(lp) if lp.exists() else 'missing', h5_sha, code_sha)
    stale = set(stale_kf_frames(out / 'kf', rkeys))
    jobs = [(k, int(fr), int(o0[k]), int(no[k]), gt_by_kf[k], pr_by_kf[k]) for k, fr in enumerate(kf_frame) if int(fr) in stale]
    if jobs:
        log(f'[viewer] {s} keyframe 그림 {len(jobs)}/{len(kf_frame)} 다시 그림 (입력 키 변경·그림 없음)')
        with Pool(workers, initializer=_init, initargs=(str(P.seq_dir(s)), str(run / 'frontend_output.h5'), str(labels), str(out / 'kf'))) as pool:
            for i, _ in enumerate(pool.imap_unordered(render_kf, jobs, chunksize=4)):
                if (i + 1) % 300 == 0:
                    log(f'[viewer] {s} keyframe 그림 {i + 1}/{len(jobs)}')
    save_render_manifest(out / 'kf', rkeys)
    H.save()

    m3, v3, a3 = s3['missed'], s3['visibility'], s3['accuracy']
    a2 = s2['all']
    el = [r for r in ep3.values() if _px(r) >= area]
    hum = lambda r: str(r.get('is_human', '0')) == '1'
    kn = lambda rs: [sum(str(r['detected']) == '1' for r in rs), len(rs)]
    geo_taus = ((geo or {}).get('params') or {}).get('taus_m') or [0.05, 0.1, 0.2]
    f_tau = int(round(geo_taus[-1] * 100))
    stamp = build_stamp(read_summary(P.RUNS / 'summary.json').get(s), P.RUNS / 'summary.json')
    data = dict(
        seq=s, seq_name=T.seq_label(s) if callable(getattr(T, 'seq_label', None)) else s,
        generated=stamp['generated'], generated_epoch=stamp['generated_epoch'], source=stamp,
        kf_label=KF_LABEL, frame_label=FRAME_LABEL, colors=COLORS,
        words=getattr(T, 'WORDS', {}) or {},
        cell=CELL, area_min_px=area, tau_cm=int(round(tau_m * 100)), f_tau_cm=f_tau,
        theta=(s2.get('params') or {}).get('theta', 0.5), labels=T.OBJECT,
        reasons=[dict(code=k, label=v) for k, v in sorted(reasons_seen.items(), key=lambda kv: (T.reason_order(kv[0]), kv[0]))],
        gt2d={k: T.gt2d_label(k) for k in GT2D_KEYS},
        pred2d={k: T.pred2d_label(k) for k in PRED2D_KEYS},
        kf3d={k: dict(label=v['label'], desc=v['desc']) for k, v in T.KF3D.items()},
        reason_desc={k: T.reason_desc(k) for k in reasons_seen},
        reason_gloss={k: (T.gloss(k) if callable(getattr(T, 'gloss', None)) else '') for k in reasons_seen},
        iou_thr=(s3.get('params') or {}).get('iou_threshold', (s3.get('params') or {}).get('min_frac', 0.5)),
        cmp_note=cmp_note(s3, s2),
        loc_min_iou=(s3.get('params') or {}).get('loc_min_iou'),
        summary=dict(
            recall_3d=v3['recall_detectable'], recall_3d_static=v3['by_kind']['static']['recall_detectable'],
            recall_3d_human=v3['by_kind']['human']['recall_detectable'],
            recall_3d_kn=kn(el), recall_3d_static_kn=kn([r for r in el if not hum(r)]), recall_3d_human_kn=kn([r for r in el if hum(r)]),
            within20=a3['within_tau_fraction_mean'],
            dup_rate=a3.get('duplicate_rate'), recall_2d=a2['recall']['0.5'] if a2['recall'] else None, n_gt_2d=a2.get('n_gt_instances'),
            pq=a2['PQ'], sq=a2['SQ'], precision_2d=a2['precision'], n_kf=len(kf_frame),
            idf1=mot['metrics']['IDF1'] if mot else None, hota=mot['metrics']['HOTA'] if mot else None,
            idsw=mot['metrics']['IDSW'] if mot else None,
            f20=geo['summary'][f'F@{f_tau}']['median'] if geo and geo.get('summary') and f'F@{f_tau}' in geo['summary'] else None),
        bg=dict(n=len(bg), xy=b64(bg), w=b64(np.minimum(cnt, 65535).astype(np.uint16))),
        pred=dict(matched=dict(n=len(pred_layers['matched']), xy=b64(pred_layers['matched'])),
                  unmatched=dict(n=len(pred_layers['unmatched']), xy=b64(pred_layers['unmatched']))),
        obj_cells=dict(n=len(obj_cells), xy=b64(obj_cells)), objects=objects,
        traj=dict(n=len(traj), xy=b64(traj)), kfs=kfs)
    data['metrics'] = metric_rows(data['summary'], data)
    (out / 'data.js').write_text('window.FB=' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';\n')
    (out / 'meta.json').write_text(json.dumps(viewer_meta(data), ensure_ascii=False, indent=1))
    shutil.copy(P.HERE / 'viewer_template.html', out / 'index.html')
    write_index(out.parent, P.RUNS / 'summary.json')                # 목록 페이지·시퀀스 전환 데이터 갱신 (m2)
    miss = [r['key'] for r in data['metrics'] if r['fallback']]
    if miss:
        log(f'[viewer] {s} 알림 — terms.py 에 이름이 없어 뷰어 기본값을 쓴 지표: {", ".join(miss)}')
    log(f'[viewer] {s} 완료 — 물체 {len(objects)} · keyframe {len(kfs)} (그림 다시 {len(jobs)}) · {time.time() - t0:.0f}s → {out / "index.html"}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seqs', nargs='*', default=P.SEQUENCES)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--index-only', action='store_true',
                    help='뷰어는 그대로 두고 목록 페이지(viewer/index.html)와 오래됨 표시만 다시 만든다 (1초 안)')
    a = ap.parse_args()
    if a.index_only:
        p = write_index(P.RUNS / 'viewer', P.RUNS / 'summary.json')
        old = [x['seq'] for x in json.loads((P.RUNS / 'viewer' / 'index.js').read_text().split('=', 1)[1].rstrip().rstrip(';'))['seqs']
               if x['stale']['state'] == 'stale']
        print(f'[viewer] 목록 갱신 → {p}' + (f' · 오래된 시퀀스 {len(old)}개: {", ".join(old)}' if old else ' · 전부 최신'))
        return
    for s in a.seqs:
        build(s, a.workers)


if __name__ == '__main__':
    main()
