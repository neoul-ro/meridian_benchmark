#!/usr/bin/env python3
"""뷰어 자체 검증 (사용성 리뷰 M1~M4 · m1~m4 · m7, 수정 ⑦).

합성 입력만 쓴다 (실제 데이터·브라우저 없음, 1초 안).
  M1  맵이 비지 않게 — 템플릿에 크기 0 방어(fit 재시도 · ResizeObserver)와 '전체 보기' 단추가 있다.
  M2  마우스 올림과 클릭이 같은 선택 함수 하나(pickAt)를 쓴다 (JS 논리 테스트는 test_viewer_js.py).
  M3  미검출 물체 목록(원인별) · 필터는 숨기지 않고 흐리게 + 개수 · keyframe 2D/3D 상태 필터용 데이터.
  M4  물체마다 (keyframe, GT) 3D 판정 행 · 과소분할/과다분할 상대 · 번호 규칙(kf 0부터 = CSV, 프레임 따로).
  m1  생성 시각 · summary.json 과 비교한 오래됨 경고.
  m2  viewer/index.html 목록 페이지 + 시퀀스 전환 (file:// 에서 fetch 없이 동작).
  m3  URL 해시 상태 (JS 쪽은 test_viewer_js.py).
  m4  범례에 같은 색이 두 뜻으로 쓰이지 않게 선 모양 구분 · 중복 문구 없음 · 맵의 채점 제외는 빗금.
  m7  지표 설명을 눌러서 여는 칸 (title 속성만으로 끝내지 않음).
  용어 모든 표시 이름은 terms.py → build_viewer.py → data.js 로만 온다 (템플릿 하드코딩 금지).
실행: python test_viewer.py
"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # frontend_benchmark/ (테스트 파일은 tests/ 에 있다)
TESTS = Path(__file__).resolve().parent            # frontend_benchmark/tests/
sys.path.insert(0, str(HERE))

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


def kfgt(kf, ep, status, **kw):
    r = dict(kf=str(kf), frame=str(int(kf) * 10), ep_index=str(ep), status=status, matched_obs='-1', best_obs='-1',
             merged_obs='-1', n_frags='0', split_cover='0.0', present='1', gt_object_id='0', best_iou='0.0')
    r.update({k: str(v) for k, v in kw.items()})
    return r


# ---------------------------------------------------------------- M4 (keyframe, GT) 판정 행
def test_m4_judgment_rows(tmp):
    """M4 물체의 3D keyframe 판정 행 — 개수는 kf3d 와 같고, 각 행에 kf·프레임·상태가 있다"""
    import build_viewer as BV
    rows = [kfgt(0, 1, 'tp', matched_obs=5), kfgt(2, 1, 'merged', merged_obs=9), kfgt(2, 3, 'tp', matched_obs=9),
            kfgt(4, 2, 'not_present'), kfgt(4, 1, 'split', n_frags=3, split_cover=0.8)]
    by_ep, by_kf = BV.kfgt_index(rows)
    js = BV.kf_judgment_rows([1, 2], by_ep, by_kf, ep_obj={1: 0, 2: 0, 3: 7})
    check('판정 행 수 = 그 물체 GT 트랙의 행 수', len(js), 4)
    check('kf 오름차순 · (kf, ep, 상태)', [(j['kf'], j['ep'], j['st']) for j in js],
          [(0, 1, 'tp'), (2, 1, 'merged'), (4, 1, 'split'), (4, 2, 'not_present')])
    check('프레임 번호를 따로 들고 있다', [j['fr'] for j in js], [0, 20, 40, 40])
    check('상태 개수 합 = object_kf3d', BV.judgment_counts(js), BV.object_kf3d([1, 2], BV.kf3d_counts(rows)))


def test_m4_counterparts(tmp):
    """M4 과소분할·과다분할의 상대 — 같은 예측에 묶인 다른 물체, 조각 예측 번호"""
    import build_viewer as BV
    rows = [kfgt(2, 1, 'merged', merged_obs=9), kfgt(2, 3, 'tp', matched_obs=9), kfgt(2, 4, 'merged', merged_obs=9),
            kfgt(2, 5, 'tp', matched_obs=8), kfgt(6, 1, 'split', n_frags=2, split_cover=0.7)]
    by_ep, by_kf = BV.kfgt_index(rows)
    js = BV.kf_judgment_rows([1], by_ep, by_kf, ep_obj={1: 0, 3: 1, 4: 2, 5: 3},
                             frags={(6, 1): [11, 12]})
    m = js[0]
    check('과소분할 상대 물체 (같은 예측 obs 9 에 묶인 것, 자기 제외)', sorted(m['mates']), [1, 2])
    check('과소분할을 만든 예측 번호', m['obs'], 9)
    check('다른 예측(obs 8)에 매칭된 물체는 상대가 아님', 3 in m['mates'], False)
    s = js[1]
    check('과다분할 조각 예측 번호', s['frags'], [11, 12])
    check('조각 개수는 CSV n_frags 를 그대로', s['nfrag'], 2)


def test_m4_frag_lookup(tmp):
    """M4 과다분할 조각 = score_observations.csv 에서 이 GT 가 주인(top_frac ≥ 0.5)인 예측"""
    import build_viewer as BV
    obs = [dict(kf='6', obs='11', top_oid='42', top_frac='0.9'), dict(kf='6', obs='12', top_oid='42', top_frac='0.6'),
           dict(kf='6', obs='13', top_oid='42', top_frac='0.3'), dict(kf='6', obs='14', top_oid='7', top_frac='1.0'),
           dict(kf='7', obs='15', top_oid='42', top_frac='0.8')]
    f = BV.frag_index(obs, {42: [1]})
    check('kf 6 에서 GT 트랙 1 의 조각', f.get((6, 1)), [11, 12])
    check('다른 keyframe 은 따로', f.get((7, 1)), [15])


# ---------------------------------------------------------------- M3 keyframe 필터 데이터
def test_m3_kf_status_counts(tmp):
    """M3 keyframe 마다 2D 상태(과다분할 따로)와 3D 상태 개수 — 필터·범례가 합산하지 않게"""
    import build_viewer as BV
    gt = [dict(status='tp'), dict(status='split'), dict(status='split'), dict(status='low_iou'), dict(status='missed')]
    check('2D 상태 개수 (합산 없음)', BV.status_counts(gt, ('tp', 'merged', 'split', 'low_iou', 'missed')),
          dict(tp=1, split=2, low_iou=1, missed=1))
    k3 = [dict(status='tp'), dict(status='merged'), dict(status='not_present')]
    check('3D 상태 개수', BV.status_counts(k3, ('tp', 'merged', 'split', 'low_iou', 'missed', 'ignored_match', 'not_present')),
          dict(tp=1, merged=1, not_present=1))


def test_m3_boxes(tmp):
    """M3·M4 라벨 그림 한 장에서 GT 트랙마다 상자 — 채점 대상이 아닌 GT 도 (이동 후 강조용)"""
    import numpy as np

    import build_viewer as BV
    lab = np.zeros((480, 640), np.uint16)
    lab[10:20, 30:50] = 1 + 1      # ep 1
    lab[100:110, 200:204] = 1 + 7  # ep 7 (작음 — 2D 채점 대상 아님)
    b = BV.kf_boxes(lab)
    check('ep 1 상자', b.get(1), [30, 10, 50, 20])
    check('작은 ep 7 도 상자가 있다', b.get(7), [200, 100, 204, 110])
    check('없는 ep', 5 in b, False)


# ---------------------------------------------------------------- m1 생성 시각 · 오래됨
def test_m1_staleness(tmp):
    """m1 뷰어가 summary.json 보다 오래됐는지 — 채점 단계 키(report_steps)로 판정"""
    import build_viewer as BV
    built = dict(steps={'score3d': 'aaa', 'seg2d': 'bbb'}, summary_mtime=100.0, generated_epoch=200.0)
    same = dict(report_steps={'score3d': dict(key='aaa'), 'seg2d': dict(key='bbb')})
    check('같은 키 → 최신', BV.staleness(built, same)['state'], 'fresh')
    diff = dict(report_steps={'score3d': dict(key='zzz'), 'seg2d': dict(key='bbb')})
    st = BV.staleness(built, diff)
    check('키가 다르면 오래됨', st['state'], 'stale')
    check('어느 단계가 다른지 알려 준다', st['steps'], ['score3d'])
    check('summary 에 이 시퀀스가 없으면 알 수 없음', BV.staleness(built, None)['state'], 'unknown')


def test_m1_generated_in_data(tmp):
    """m1 data.js 에 생성 시각(사람이 읽는 글 + epoch)과 summary 출처가 들어간다"""
    import build_viewer as BV
    d = BV.build_stamp(summary={'report_steps': {'score3d': dict(key='k1', finished='2026-09-18 01:43:18')}},
                       summary_path=Path(tmp) / 'nope.json')
    check('생성 시각 글', isinstance(d['generated'], str) and len(d['generated']) >= 16, True)
    check('생성 시각 epoch', isinstance(d['generated_epoch'], float), True)
    check('채점 단계 키를 기록', d['steps'], {'score3d': 'k1'})
    check('채점 끝난 시각도 기록', d['scored'], '2026-09-18 01:43:18')


# ---------------------------------------------------------------- m2 목록 페이지
def test_m2_index_page(tmp):
    """m2 viewer/index.html — 시퀀스 목록 + 대표 숫자, 서버 없이(file://) 열린다"""
    import build_viewer as BV
    v = Path(tmp) / 'viewer'
    for s, rc in (('apartment_s1_00h', 0.8), ('office_s1_06h', 0.9)):
        d = v / f'uHumans2_{s}'
        d.mkdir(parents=True)
        (d / 'meta.json').write_text(json.dumps(dict(
            seq=s, dir=d.name, generated='2026-09-18 03:00', generated_epoch=1.0,
            stats=[dict(key='recall_3d', label='트랙 재현율', value=f'{rc:.0%}')],
            source=dict(steps={'score3d': 'k1'}, summary_mtime=0.0))))
    summ = v.parent / 'summary.json'
    summ.write_text(json.dumps({'apartment_s1_00h': dict(report_steps={'score3d': dict(key='k1')}),
                                'office_s1_06h': dict(report_steps={'score3d': dict(key='k2')})}))
    p = BV.write_index(v, summ)
    html = (v / 'index.html').read_text()
    js = (v / 'index.js').read_text()
    check('index.html 을 만든다', p == v / 'index.html' and (v / 'index.html').exists(), True)
    check('fetch·XMLHttpRequest 를 쓰지 않는다 (file:// 에서 막힘)', ('fetch(' in html) or ('XMLHttpRequest' in html), False)
    check('index.js 를 <script> 로 읽는다', 'index.js' in html, True)
    check('두 시퀀스 모두 링크', all(f'uHumans2_{s}/index.html' in js or f'uHumans2_{s}/index.html' in html
                             for s in ('apartment_s1_00h', 'office_s1_06h')), True)
    check('대표 숫자가 들어 있다', '80%' in js, True)
    data = json.loads(js[js.index('=') + 1:].rstrip().rstrip(';'))
    st = {x['seq']: x['stale']['state'] for x in data['seqs']}
    check('키가 같은 시퀀스는 최신, 다른 시퀀스는 오래됨', st, {'apartment_s1_00h': 'fresh', 'office_s1_06h': 'stale'})
    check('뷰어가 시퀀스 전환에 쓸 목록', [x['dir'] for x in data['seqs']],
          ['uHumans2_apartment_s1_00h', 'uHumans2_office_s1_06h'])


# ---------------------------------------------------------------- 용어 (terms.py 한 곳)
def test_terms_only_labels(tmp):
    """용어 표시 이름은 전부 data.js 를 거친다 — 템플릿에 상태·원인·지표 이름을 하드코딩하지 않는다"""
    import terms as T
    html = (HERE / 'viewer_template.html').read_text()
    # 4글자 이상인 표시 이름만 본다 ('TP' · '검출' 같은 짧은 말은 '예측 검출' 처럼 일반 문장에도 들어간다)
    hard = sorted({w for w in ([r['label'] for r in T.REASONS.values()] + [v['label'] for v in T.KF3D.values()] +
                               list(T.GT2D.values()) + list(T.PRED2D.values()) + list(T.OBJECT.values()))
                   if len(w) >= 4 and w in html})
    check('템플릿에 하드코딩된 표시 이름 없음', hard, [])
    check('물체 판정 이름표를 data 에서 받음', 'D.labels' in html, True)
    check('2D·3D 상태 이름표를 data 에서 받음', ('D.gt2d' in html and 'D.kf3d' in html), True)


def test_metric_labels_from_terms(tmp):
    """용어 머리말 지표 이름·설명도 terms.py 에서. 없으면 눈에 띄게 대체하고 그 사실을 data 에 남긴다"""
    import build_viewer as BV
    S = dict(recall_3d=0.8082, recall_3d_kn=[198, 245], recall_3d_static=0.8, recall_3d_human=None,
             recall_3d_static_kn=[190, 237], recall_3d_human_kn=[8, 8], within20=0.9295, f20=0.92, recall_2d=0.558,
             pq=0.4226, sq=0.8, precision_2d=0.5, idf1=0.3966, hota=0.441, idsw=215, n_kf=204, dup_rate=0.07,
             n_gt_2d=2366)
    ctx = dict(area_min_px=1600, tau_cm=20, f_tau_cm=20, theta=0.5, iou_thr=0.5)
    rows = BV.metric_rows(S, ctx)
    keys = [r['key'] for r in rows]
    check('머리말 지표 행', keys[:4], ['recall_3d', 'recall_3d_kind', 'within_tau', 'f_tau'])
    check('값은 문자열로 미리 만든다', rows[0]['value'], '80.8%')
    check('개수에서 한 번만 반올림 (198/245)', BV.pct_kn([198, 245]), '80.8%')
    check('설명이 있다 (m7 눌러서 여는 칸)', all(len(r['desc']) > 10 for r in rows), True)
    check('terms 에 없는 이름은 표시로 남긴다', all('fallback' in r for r in rows), True)


# ---------------------------------------------------------------- 템플릿 (M1 · M2 · M3 · m3 · m4 · m7)
def test_template_features(tmp):
    """M1·M2·m3·m4·m7 템플릿에 방어와 조작 장치가 있다"""
    html = (HERE / 'viewer_template.html').read_text()
    check('M1 크기 0 방어 (ResizeObserver)', 'ResizeObserver' in html, True)
    check('M1 배율이 0 이면 다시 맞춤', 'needFit' in html or 'validView' in html, True)
    check('M1·m3 전체 보기(reset) 단추', 'id="reset"' in html, True)
    check('M2 선택 함수 하나 (pickAt 호출 지점 1곳 · 올림과 클릭이 같은 함수)',
          html.count('LIB.pickAt(') == 1 and html.count('pickEvent(e)') >= 2, True)
    check('m3 URL 해시 상태', 'parseHash' in html and 'hashchange' in html, True)
    check('m7 설명을 눌러서 열기 (aria-expanded)', 'aria-expanded' in html, True)
    check('m2 시퀀스 전환 select', 'id="seqsel"' in html, True)
    check('m1 오래됨 경고 자리', 'id="stale"' in html, True)
    check('M3 미검출 목록 자리', 'missList' in html, True)
    check('M4 추적 띠 칸을 누를 수 있다', 'data-kfcell' in html, True)
    check('JS 논리 묶음 (node 테스트용)', 'id="fblib"' in html, True)
    check('m4 목록 줄 간격 규칙', '.inst button span' in html or '.inst .row' in html, True)
    check('m4 맵에서 채점 제외를 빗금으로 (색 말고 무늬)', 'LIB.hatch(' in html and 'function hatch(' in html, True)
    check('2차 keyframe 그림을 자리에 맞춰 키움 (+ 확대 단추)', 'layoutKf' in html and 'id="zin"' in html, True)
    check('2차 고른 물체 정보가 옆 패널 맨 위', 'state.sel >= 0 ? objectPanel' in html, True)
    check('2차 축소했을 때 미검출 고리', 'markSmall' in html, True)
    check('2차 글자 크기 11px 이하 없음', 'font-size: 11px' in html or 'font-size: 10px' in html, False)
    check('2차 summary.json 은 읽을 수 있을 때만 요청', 'canReadSummary' in html, True)
    check('m4 범례 무늬가 맵과 같은 규칙', "'stripe' : 'fill'" in html, True)


def test_m4_2d_3d_conflict(tmp):
    """M4 같은 (keyframe, GT) 의 2D 판정을 함께 담고, 둘이 다를 수 있는 이유를 params 에서 만든다"""
    import build_viewer as BV
    rows = [kfgt(4, 1, 'split', n_frags=3), kfgt(5, 1, 'tp', matched_obs=9)]
    by_ep, by_kf = BV.kfgt_index(rows)
    st2 = BV.gt2d_index([dict(kf='4', ep_index='1', status='missed', eligible='True', best_iou='0.04'),
                         dict(kf='5', ep_index='1', status='tp', eligible='False', best_iou='0.9')])
    js = BV.kf_judgment_rows([1], by_ep, by_kf, frags={(4, 1): [11, 12, 13]}, st2=st2)
    check('3D 과다분할 행에 2D 상태와 IoU', (js[0]['st2'], js[0]['iou2']), ('missed', 0.04))
    check('2D 채점 대상이 아니면 상태를 비운다', (js[1]['st2'], js[1]['iou2']), ('', -1))
    note = BV.cmp_note(dict(params=dict(tau_m=0.2, voxel_m=0.02, max_range_m=5.0, iou_threshold=0.5)),
                       dict(params=dict(theta=0.5)))
    for w in ('20cm', 'IoU_τ ≥ 0.5', '2cm', '5m', 'IoU > 0.5', 'void'):
        check(f'설명에 {w}', w in note, True)
    check('숫자를 params 에서 만든다 (τ 10cm 면 10cm 로)',
          '10cm' in BV.cmp_note(dict(params=dict(tau_m=0.1)), dict(params=dict(theta=0.25))), True)
    html = (HERE / 'viewer_template.html').read_text()
    check('템플릿이 설명을 data 에서 받음', 'D.cmp_note' in html, True)
    check('조각 예측을 이미지에 표시 (점선 상자)', 'markPreds' in html and 'pboxes' in html, True)


def test_n7_no_duplicate_desc(tmp):
    """N7 머리말 설명 칸에 같은 문장이 두 번 나오지 않는다 (terms 의 쉬운 뜻 + 뷰어의 조건)"""
    import build_viewer as BV
    check('같은 말이면 한 번만', BV.join_desc('예측 복셀 중 20cm 이내 비율의 평균', '예측 복셀 중 20cm 이내인 비율의 평균'),
          '예측 복셀 중 20cm 이내 비율의 평균')
    check('다른 말이면 이어 붙임', BV.join_desc('쉬운 뜻', '분모는 매칭된 예측 검출'), '쉬운 뜻 · 분모는 매칭된 예측 검출')
    check('terms 설명이 없으면 그대로', BV.join_desc('', '뷰어 설명'), '뷰어 설명')
    S = dict(recall_3d=0.8, recall_3d_kn=[198, 245], recall_3d_static=0.8, recall_3d_human=None, recall_3d_static_kn=[190, 237],
             recall_3d_human_kn=[8, 8], within20=0.93, f20=0.92, recall_2d=0.55, pq=0.42, sq=0.8, precision_2d=0.5,
             idf1=0.4, hota=0.44, idsw=1, n_kf=204, dup_rate=0.07, n_gt_2d=2366)
    rows = BV.metric_rows(S, dict(area_min_px=1600, tau_cm=20, f_tau_cm=20, theta=0.5, iou_thr=0.5))
    dup = []
    for r in rows:
        parts = [s.strip() for s in r['desc'].split(' · ') if s.strip()]
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                if BV.same_sentence(a, b):
                    dup.append((r['key'], a, b))
    check('설명 안에 겹치는 문장 없음', dup, [])


def test_numbering_convention(tmp):
    """M4 번호 규칙 — kf 는 CSV 의 kf 열(0부터), 프레임은 데이터셋 프레임. 둘 다 이름을 붙인다"""
    import build_viewer as BV
    check('kf 라벨', BV.KF_LABEL, 'kf')
    check('프레임 라벨', BV.FRAME_LABEL, '프레임')
    check('한 줄 표기', BV.kf_text(239, 2386), 'kf 239 · 프레임 2386')
    html = (HERE / 'viewer_template.html').read_text()
    check('템플릿이 번호 규칙을 data 에서 받음', 'D.kf_label' in html and 'D.frame_label' in html, True)


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for fn in (test_m4_judgment_rows, test_m4_counterparts, test_m4_frag_lookup, test_m3_kf_status_counts,
                   test_m3_boxes, test_m1_staleness, test_m1_generated_in_data, test_m2_index_page,
                   test_terms_only_labels, test_metric_labels_from_terms, test_template_features,
                   test_m4_2d_3d_conflict, test_n7_no_duplicate_desc, test_numbering_convention):
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
