"""사용자에게 보이는 용어를 한 곳에서 정한다 (감사 D_terms 결정 · 사용성 리뷰 M5). JSON·CSV 의 키와 값은 바꾸지 않고, 표시할 때만 여기서 바꾼다.

report.py(summary.md) · build_viewer.py(뷰어) · status_md.py · compare.py · render_examples.py 가 같은 표를 쓴다.

지표 (METRICS) — 이름 · 모집단 · 단위 · 소수 자리수 · 쉬운 뜻 · 좋은 방향이 여기 한 곳에 있다 (M5).
  이름에는 모집단을 넣는다: '트랙 재현율 (Easy)' ≠ '트랙 재현율 (전체 채점 대상)'.
  같은 종류(kind)는 어디서나 같은 형식으로 쓴다 — percent 1자리 · score 2자리 · cm 2자리 · per_track 1자리 · count 정수.
  표시는 fmt(key, value) 한 곳을 지난다. summary.md · status.md · 뷰어 · compare 가 같은 글자를 쓴다.
말 고르기 (M5 '한 낱말 = 한 뜻')
  과다분할(split) · 과소분할(merged) 로만 쓴다 — 같은 뜻의 다른 낱말을 섞지 않는다 (사용성 리뷰 M5).
  '제외' 는 '채점 제외(ineligible) = 채점에 넣지 않는 GT' 한 가지 뜻으로만 쓴다.
  present 아닌 GT 에 매칭된 예측은 '무시 GT 매칭', void 에 떨어진 마스크는 'void 겹침 무시', 거리 밖 점은 '범위 밖 제거' 다.
  accuracy 는 기준이 둘이라 이름으로 구분한다 — 'GT 트랙 점군 기준'(2장) · '같은 keyframe GT 표면 기준'(기하).
미검출 원인 (score_episodes.csv 의 miss_reason 값) — 3D 매칭 = keyframe 별 허용 거리 IoU_τ ≥ 0.5 (26/09/17 결정 ④, 수정 ④ 구현)
  no_kf / merged / split / low_iou / missed. 판정 순서 merged > split > low_iou > missed, 트랙은 최빈값 (match3d.py).
  26/09/17 이전 결과의 옛 값은 표에서 뺐다 — 옛 결과를 읽으면 '알 수 없는 이유 ‹값›' 으로 보인다(조용히 버리지 않음).
  옛 값 대응은 score.json params.reason_mapping 에 남아 있다. 값 이름이 바뀌면 이 표 한 곳만 고친다.
(keyframe, GT) 3D 상태 (score_kf_gt.csv 의 status) — KF3D
TIDE 오류 (score_observations.csv 의 tide_error, 매칭 안 된 예측) — TIDE_ERR
색 (COLORS) — 뷰어 · 판정 그림이 같은 색을 쓴다 (dataviz 검증: 색약 포함 모든 쌍 구분)
"""
from decimal import ROUND_HALF_UP, Decimal

import tide_rules as TR

UNKNOWN = '알 수 없는 이유'
_LOC = f'{TR.LOC_MIN_IOU:g}'

# group: 뷰어 물체 색 분류 (merged = 과소분할 색, missed = 미검출 색) · order: 동률일 때 앞선 것 (match3d.STATUS_ORDER 와 같은 순서)
REASONS = {
    'merged': dict(label='과소분할(under-segmentation)', group='merged', order=0,
                   desc='한 예측 검출이 이 GT 와 다른 GT 를 각각 절반 이상 배타적으로 덮음 (덮음 = 가장 가까운 GT 소유, 20cm 허용)'),
    'split': dict(label='과다분할(over-segmentation)', group='missed', order=1,
                  desc='복셀 절반 이상이 이 GT 소유인 예측 검출이 2개 이상이고, 그 조각들이 이 GT 를 절반 이상 배타적으로 덮음'),
    'low_iou': dict(label='Loc(localization error)', group='missed', order=2,
                    desc=f'매칭 안 됐고 가장 잘 맞는 예측 검출의 IoU_τ 가 {_LOC} 이상 0.5 이하 (TIDE Loc 범위, 2D 와 같은 규칙)'),
    'no_kf': dict(label='keyframe 없음', group='missed', order=3,
                  desc='GT 트랙이 있음(present: 그 keyframe 라벨 5m 안 1600px 이상)인 publish(발행)된 keyframe 이 0개'),
    'missed': dict(label='기타 미검출(Miss)', group='missed', order=4, desc='present 인 keyframe 이 있었지만 위 원인 어디에도 해당하지 않음'),
}

# (keyframe, GT) 3D 상태 — score_kf_gt.csv status. 앞 5개는 present(라벨 ≥ 1600px) 쌍, 뒤 2개는 present 아닌 쌍
KF3D = {
    'tp': dict(label='TP', desc='present 이고 IoU_τ ≥ 0.5 로 매칭'),
    'merged': dict(label='과소분할(under-segmentation)', desc=REASONS['merged']['desc']),
    'split': dict(label='과다분할(over-segmentation)', desc=REASONS['split']['desc']),
    'low_iou': dict(label='Loc(localization error)', desc=REASONS['low_iou']['desc']),
    'missed': dict(label='미검출(Miss)', desc='present 인데 위 어디에도 해당 안 됨'),
    'ignored_match': dict(label='무시 GT 매칭(ignored match)', desc='present 아닌 GT 에 매칭 — TP 도 FP 도 아님'),
    'not_present': dict(label='present 아님', desc='라벨 표면은 있지만 5m 안 라벨 1600px 미만이고 매칭도 없음 — 채점 안 함'),
}

# 매칭 안 된 예측의 TIDE 오류 — score_observations.csv tide_error (tide_rules.pred_error)
TIDE_ERR = {
    'dupe': dict(label='중복(Dupe)', desc='다른 예측 검출이 이미 매칭한 present GT 와 IoU_τ ≥ 0.5 (TIDE quantify.py:250-255)'),
    'loc': dict(label='Loc', desc=f'최대 IoU_τ 가 {_LOC} 이상 0.5 이하 (TIDE quantify.py:236-240)'),
    'bkg': dict(label='배경(Bkg)', desc=f'present GT 와 최대 IoU_τ 가 {_LOC} 미만 (TIDE quantify.py:258-262)'),
    'other': dict(label='기타(Other)', desc='위 어디에도 해당 안 됨 (TIDE quantify.py:264-265)'),
}

# 2D GT 상태 (score_2d_gt.csv status) · 예측 상태 (score_2d_pred.csv status)
GT2D = {
    'tp': 'TP',
    'merged': '과소분할(under-segmentation)',
    'split': '과다분할(over-segmentation)',
    'low_iou': 'Loc(localization error)',
    'missed': '미검출(Miss)',
}
PRED2D = {
    'tp': 'TP',
    'tp_small': '무시 GT 매칭(ignored match)',
    'ignore': 'void 겹침 무시',
    'fp': '오검출(FP)',
}

# 뷰어 물체 판정
OBJECT = {
    'detected': '검출',
    'missed': '미검출',
    'merged': '과소분할(under-segmentation)',
    'ineligible': '채점 제외(ineligible)',
}

# 뷰어 · 판정 그림 공통 색 (viewer_template.html 의 C 와 같은 값). 배경이 어두울 때 대비 3:1 이상
COLORS = {
    'detected': '#3987e5',
    'missed': '#d95926',
    'merged': '#199e70',
    'ineligible': '#898781',
}

# 한 낱말 = 한 뜻 (M5). 화면에 쓰는 낱말과 그 뜻 — 문서·코드가 같은 말을 쓰게 여기서 고른다
WORDS = {
    'ineligible': '채점 제외(ineligible) — 채점 대상이 아닌 GT (가시 픽셀이 기준 미만)',
    'ignored_match': '무시 GT 매칭 — present 아닌 GT 에 매칭된 예측 검출 (TP 도 FP 도 아님)',
    'void_overlap': 'void 겹침 무시 — 절반 넘게 void 에 떨어져 채점에서 뺀 예측 마스크',
    'out_of_range': '범위 밖 제거 — 평가 거리 범위(5m) 밖이라 뺀 점',
    'no_level': '등급 없음 — 난이도 기준 밖이라 Easy/Moderate/Hard 어디에도 안 드는 인스턴스',
}

# 짧은 뜻풀이 (한 번만 보여 주는 쉬운 말)
GLOSS = {
    'merged': '여러 물체를 하나로 뭉쳐 본 것',
    'split': '한 물체를 여러 조각으로 나눠 본 것',
    'low_iou': '자리는 잡았는데 많이 어긋난 것',
    'no_kf': '그 물체가 크게 보인 keyframe 이 발행되지 않은 것',
    'missed': '아예 못 본 것',
}

# ---------------------------------------------------------------- 지표 (M5: 이름 · 모집단 · 단위 · 자리수 · 방향)
# kind: percent(비율 → %) · score(0~1 점수) · cm(거리) · per_track · count.  higher: True 면 클수록 좋다
DECIMALS = {'percent': 1, 'score': 2, 'cm': 2, 'per_track': 1, 'count': 0}
_M = lambda label, kind, population='', gloss='', higher=True: dict(
    label=label, kind=kind, population=population, gloss=gloss, higher=higher, decimals=DECIMALS[kind],
    unit={'percent': '%', 'cm': 'cm'}.get(kind, ''))

METRICS = {
    'track_recall_easy': _M('트랙 재현율', 'percent', 'Easy', '크고 잘 보인 물체를 한 번이라도 잡은 비율'),
    'track_recall_moderate': _M('트랙 재현율', 'percent', 'Moderate', 'Easy + 조금 더 어려운 것까지'),
    'track_recall_hard': _M('트랙 재현율', 'percent', 'Hard', 'Easy + Moderate + 더 어려운 것까지'),
    'track_recall_eligible': _M('트랙 재현율', 'percent', '전체 채점 대상', '채점 대상 물체를 한 번이라도 잡은 비율'),
    'track_recall_in_view': _M('트랙 재현율', 'percent', '가시', '한 번이라도 화면에 보인 물체 기준'),
    'track_recall_all': _M('트랙 재현율', 'percent', '전체 GT 트랙', '작아서 채점 대상이 아닌 것까지 포함한 기준'),
    'track_recall_static': _M('트랙 재현율', 'percent', '채점 대상 · 정적 물체', '사람을 뺀 물체만 본 트랙 재현율'),
    'track_recall_human': _M('트랙 재현율', 'percent', '채점 대상 · 사람', '사람 GT 트랙만 본 트랙 재현율 (없는 시퀀스는 –)'),
    'det_recall': _M('검출 재현율 (DetRe)', 'percent', 'present 쌍', '보이던 순간마다 그 keyframe 에서 잡았는지'),
    # τ · IoU 임계가 이름에 들어가는 지표는 기본값(τ = 20cm · IoU > 0.5)으로 적어 둔다. 실행 params 가 다르면 부르는 쪽에서 라벨을 만든다
    'within_tau': _M('P@20cm', 'percent', '매칭된 예측 검출', '예측 복셀 중 그 GT 트랙 점군 20cm 이내인 비율의 평균'),
    'recall_2d': _M('2D 재현율 @IoU0.5', 'percent', '2D 채점 대상 GT 인스턴스', '이미지에서 GT 인스턴스를 맞춘 비율'),
    'n_kf': _M('keyframe 수', 'count', 'publish 된 keyframe', 'frontend 가 발행한 keyframe 수 = 추적의 timestep 수'),
    'obs_match_rate': _M('매칭률', 'percent', '전체 예측 검출', 'GT 에 붙은 예측 검출의 비율 (정밀도가 아님)'),
    'duplicate_rate': _M('중복 검출 비율', 'percent', '전체 예측 검출', '같은 GT 를 두 번 이상 낸 비율', higher=False),
    'f20': _M('F@20cm', 'score', '매칭된 예측 검출', '표면이 20cm 안에서 GT 와 겹치는 정도 (1이 만점)'),
    'pq': _M('PQ', 'score', '2D 채점 대상 GT 인스턴스', '분할 품질 — 겹침 정확도 × 맞춘 비율 (1이 만점)'),
    'sq': _M('SQ', 'score', '매칭 쌍', '매칭된 쌍의 평균 겹침'),
    'rq': _M('RQ', 'score', '2D 채점 대상 GT 인스턴스', '맞춘 비율 (F1)'),
    'idf1': _M('IDF1', 'score', '정적 GT 트랙', '같은 물체에 같은 ID 를 계속 붙였는지 (1이 만점)'),
    'hota_alpha': _M('HOTA_α', 'score', '정적 GT 트랙', '검출과 ID 유지를 함께 본 점수 (α 한 점, 1이 만점)'),
    'deta_alpha': _M('DetA_α', 'score', '정적 GT 트랙', 'HOTA 의 검출 쪽'),
    'assa_alpha': _M('AssA_α', 'score', '정적 GT 트랙', 'HOTA 의 ID 유지 쪽'),
    'mota': _M('MOTA', 'score', '정적 GT 트랙',
               '놓친 것 · 잘못 낸 것 · ID 바뀜을 GT 검출 수로 나눠 1에서 뺀 값 — 1이 만점이고 잘못 낸 것이 많으면 음수도 나온다'),
    'ids_per_track': _M('GT 트랙당 예측 ID 수', 'per_track', '한 번 이상 매칭된 GT 트랙', 'ID 가 몇 번이나 바뀌는지', higher=False),
    'accuracy_track_cm': _M('accuracy (GT 트랙 점군 기준)', 'cm', '매칭된 예측 검출',
                            '예측 점에서 그 물체의 GT 점군까지 평균 거리', higher=False),
    'accuracy_keyframe_cm': _M('accuracy (같은 keyframe GT 표면 기준)', 'cm', '매칭된 예측 검출',
                               '예측 점에서 그 keyframe 에 실제로 보인 GT 표면까지 평균 거리', higher=False),
    'completeness_cm': _M('completeness (같은 keyframe GT 표면 기준)', 'cm', '매칭된 예측 검출',
                          'GT 표면에서 예측까지 평균 거리', higher=False),
    'chamfer_cm': _M('Chamfer-L1 (같은 keyframe GT 표면 기준)', 'cm', '매칭된 예측 검출',
                     'accuracy 와 completeness 의 평균', higher=False),
    'n_merged': _M('과소분할 건수', 'count', '2D 매칭 안 된 GT 인스턴스', '여러 물체를 하나로 뭉쳐 본 건수', higher=False),
    'n_split': _M('과다분할 건수', 'count', '2D 매칭 안 된 GT 인스턴스', '한 물체를 여러 조각으로 나눠 본 건수', higher=False),
    'idsw': _M('IDSW', 'count', '정적 GT 트랙', 'ID 가 바뀐 횟수', higher=False),
}


def seq_label(s):
    """시퀀스 표시 이름 — 화면에서는 어디서나 짧은 쪽 하나로 (apartment_s1_00h → apartment_00h). 폴더 이름은 uHumans2_<시퀀스>."""
    return str(s).replace('uHumans2_', '').replace('_s1_', '_')


def metric(key):
    return METRICS.get(key) or {}


def metric_label(key, population=True):
    """지표 표시 이름. population=True 면 모집단까지 — '트랙 재현율 (Easy)'."""
    m = METRICS.get(key)
    if not m:
        return _unknown(key)
    return f'{m["label"]} ({m["population"]})' if population and m['population'] else m['label']


def metric_gloss(key):
    return (METRICS.get(key) or {}).get('gloss', '')


def metric_higher_better(key):
    return bool((METRICS.get(key) or {}).get('higher', True))


def gloss(code):
    """미검출 원인 · 상태 값의 짧은 뜻풀이 (한 번만 보여 주는 쉬운 말)."""
    return GLOSS.get(code, '')


def _q(x, nd):
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)


def fmt(key, value, dash='–'):
    """지표 값 → 화면 문자열. 종류마다 자리수·단위가 하나다 (M5). percent 는 0~1 비율을 받는다."""
    if value is None:
        return dash
    m = METRICS.get(key)
    if not m:
        return str(value)
    nd, kind = m['decimals'], m['kind']
    if kind == 'percent':
        return f'{_q(Decimal(repr(float(value))) * 100, nd)}%'
    if kind == 'count':
        return f'{int(round(float(value)))}'
    return f'{_q(value, nd)}{m["unit"]}'


def fmt_by_kind(kind, value, dash='–', unit=True):
    """지표 key 가 없을 때 종류만으로 형식을 맞춘다 (score 2자리 · cm 2자리 · percent 1자리 · per_track 1자리 · count 정수).
    unit=False 는 표의 열 이름에 이미 단위가 있을 때 (같은 단위를 두 번 쓰지 않는다)."""
    if value is None:
        return dash
    nd = DECIMALS[kind]
    if kind == 'percent':
        return f'{_q(Decimal(repr(float(value))) * 100, nd)}%' if unit else f'{_q(Decimal(repr(float(value))) * 100, nd)}'
    if kind == 'count':
        return f'{int(round(float(value)))}'
    return f'{_q(value, nd)}' + ({'cm': 'cm'}.get(kind, '') if unit else '')


def fmt_kn(key, k, n, dash='–'):
    """개수에서 한 번만 반올림해 비율을 만든다 (감사 C15). percent 지표 전용."""
    if not n or k is None:
        return dash
    nd = metric(key).get('decimals', 1)
    return f'{(Decimal(100 * int(k)) / Decimal(int(n))).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)}%'


def _unknown(v):
    return f'{UNKNOWN} ‹{v}›'


def reason_label(code):
    r = REASONS.get(code)
    return r['label'] if r else _unknown(code)


def reason_group(code):
    r = REASONS.get(code)
    return r['group'] if r else 'missed'


def reason_order(code):
    r = REASONS.get(code)
    return r['order'] if r else 99


def reason_desc(code):
    r = REASONS.get(code)
    return r['desc'] if r else '표(terms.py REASONS)에 없는 값 — 채점기 출력이 바뀌었는지 확인할 것'


def pick_reason(codes):
    """여러 GT 트랙(에피소드)의 미검출 원인 → 대표 원인 하나. 규칙: 가장 많은 값, 동률이면 order 가 작은 값, 그다음 값 이름순.
    뷰어의 색(reason_group)과 이유 글이 모두 이 결과 하나에서 나온다 (감사 C12)."""
    codes = [c for c in codes if c not in (None, '')]
    if not codes:
        return ''
    cnt = {}
    for c in codes:
        cnt[c] = cnt.get(c, 0) + 1
    return min(cnt, key=lambda c: (-cnt[c], reason_order(c), c))


def kf3d_label(st):
    r = KF3D.get(st)
    return r['label'] if r else _unknown(st)


def tide_label(e):
    r = TIDE_ERR.get(e)
    return r['label'] if r else _unknown(e)


def gt2d_label(st):
    return GT2D.get(st, _unknown(st))


def pred2d_label(st):
    return PRED2D.get(st, _unknown(st))


def reasons_text(d, sep=', '):
    """{원인 값: 개수} → '과소분할(under-segmentation) 16, 기타 미검출(Miss) 22'. 같은 표시 이름끼리는 합친다. 빈 dict → '–'."""
    merged = {}
    order = {}
    for k, n in (d or {}).items():
        lab = reason_label(k)
        merged[lab] = merged.get(lab, 0) + int(n)
        order[lab] = min(order.get(lab, 99), reason_order(k))
    return sep.join(f'{lab} {merged[lab]}' for lab in sorted(merged, key=lambda x: (order[x], x))) or '–'
