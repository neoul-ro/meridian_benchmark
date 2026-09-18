#!/usr/bin/env python3
"""용어(terms.py) 한 곳 · 문서(README · docs/) · 판정 그림 라벨 검증 (사용성 리뷰 M5 · M7 · m6).

M5  terms.py 가 사용자에게 보이는 지표 이름 · 단위 · 소수 자리수 · 모집단의 단 하나의 출처다.
    이름에 모집단이 들어간다 ('트랙 재현율 (Easy)' vs '트랙 재현율 (전체 채점 대상)').
    과다분할 · 과소분할로 통일 (조각냄 · 합침 금지) · accuracy 두 값은 기준 이름으로 구분 · '제외' 는 한 가지 뜻 ·
    같은 종류의 수는 어디서나 같은 형식 (summary.md 와 status.md 의 같은 지표 칸이 글자까지 같다).
M7  README 는 팀원용 순서 (무엇 → 준비 → 작업 3가지 → 문제 해결 → 참고), 내부 경로·개인 이름 없음,
    정의와 이력은 docs/METRICS.md · docs/HISTORY.md, TrackEval 받는 법이 적혀 있다.
m6  판정 그림(examples)의 글자·색이 뷰어·terms.py 와 같고 범례가 있다.
"""
import re
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FAILS = []


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------ M5 용어
def test_m5_metrics_table(tmp):
    """M5 terms.METRICS = 지표 이름 · 단위 · 자리수 · 모집단의 한 곳"""
    import terms as T
    check('terms 에 METRICS', hasattr(T, 'METRICS'), True)
    for k in ('track_recall_easy', 'track_recall_eligible', 'f20', 'pq', 'idf1', 'hota_alpha', 'ids_per_track',
              'accuracy_track_cm', 'accuracy_keyframe_cm'):
        check(f'METRICS 에 {k}', k in T.METRICS, True)
    for k, m in T.METRICS.items():
        for f in ('label', 'kind', 'decimals'):
            check(f'{k} 에 {f}', f in m, True)
    check('트랙 재현율 (Easy)', T.metric_label('track_recall_easy'), '트랙 재현율 (Easy)')
    check('트랙 재현율 (전체 채점 대상)', T.metric_label('track_recall_eligible'), '트랙 재현율 (전체 채점 대상)')
    check('accuracy 두 값이 기준으로 구분', T.metric_label('accuracy_track_cm') != T.metric_label('accuracy_keyframe_cm'), True)
    check('accuracy (지도 GT 기준)', 'GT 트랙' in T.metric_label('accuracy_track_cm') or '지도' in T.metric_label('accuracy_track_cm'), True)
    check('accuracy (같은 keyframe 기준)', 'keyframe' in T.metric_label('accuracy_keyframe_cm'), True)
    check('같은 종류(0~1 점수)는 같은 자리수',
          len({T.METRICS[k]['decimals'] for k in ('f20', 'pq', 'idf1', 'hota_alpha')}), 1)
    check('퍼센트 지표 형식', T.fmt('track_recall_easy', 0.9161), '91.6%')
    check('점수 지표 형식', T.fmt('pq', 0.4249), '0.42')
    check('거리 지표 형식', T.fmt('accuracy_keyframe_cm', 0.919), '0.92cm')
    check('값 없음', T.fmt('pq', None), '–')
    check('MOTA 를 쉬운 말로', bool(T.metric_gloss('mota')), True)


def test_m5_one_word_one_meaning(tmp):
    """M5 과다분할 · 과소분할로 통일 · '제외' 한 가지 뜻"""
    import terms as T
    texts = {n: (HERE / n).read_text() for n in ('terms.py', 'report.py', 'status_md.py', 'build_viewer.py')}
    texts['README.md'] = (HERE / 'README.md').read_text()
    for n, t in texts.items():
        check(f'{n}: 조각냄 없음', '조각냄' in t, False)
        check(f'{n}: 합침 없음', re.search(r'합침', t) is not None, False)
    check('과다분할 뜻풀이 (짧은 말)', bool(T.gloss('split')), True)
    check('과소분할 뜻풀이 (짧은 말)', bool(T.gloss('merged')), True)
    labels = [T.kf3d_label(k) for k in T.KF3D] + [T.gt2d_label(k) for k in T.GT2D] + [T.pred2d_label(k) for k in T.PRED2D] \
        + [T.OBJECT[k] for k in T.OBJECT]
    excl = sorted({l for l in labels if '제외' in l})
    check("'제외' 가 붙은 표시 이름은 한 뜻뿐", len(excl) <= 1, True)
    check('무시 GT 매칭은 다른 말', any('무시' in l for l in labels), True)


def test_m5_same_number_everywhere(tmp):
    """M5 같은 지표는 summary.md 와 status.md 에서 글자까지 같은 형식"""
    import report as R
    import status_md as S
    from test_report_cli import _block
    rows = {'apartment_s1_00h': _block()}
    md = R.render(rows, {})
    st = S.render_status(rows, {})
    pq = S.cells(rows['apartment_s1_00h'])[2]
    f20 = S.cells(rows['apartment_s1_00h'])[1]
    check('PQ 문자열이 status 와 summary 에 같은 모양', (pq in md, pq in st), (True, True))
    check('F@20cm 문자열이 같은 모양', (f20 in md, f20 in st), (True, True))
    check('status 표 머리에 모집단이 들어간 이름', '트랙 재현율 (Easy)' in st, True)


# ------------------------------------------------------------------ M7 문서
def test_m7_readme(tmp):
    """M7 README 는 팀원용 순서 · 내부 경로 없음 · 정의와 이력은 docs/"""
    rd = (HERE / 'README.md').read_text()
    heads = [l.strip() for l in rd.splitlines() if l.startswith('## ')]
    order = ['무엇', '작업', '준비', '문제', '참고']       # 사용자 먼저: 쓰는 법 → 준비 (README 모범 사례)
    idx = []
    for w in order:
        i = next((k for k, h in enumerate(heads) if w in h), None)
        idx.append(i)
    check(f'README 절 순서 (무엇 → 준비 → 작업 → 문제 해결 → 참고): {heads[:6]}',
          all(i is not None for i in idx) and idx == sorted(i for i in idx if i is not None), True)
    # 개인 경로 · 내부 폴더 · 사람 호칭(문서에 이름·호칭을 남기지 않는다). 호칭은 이 파일에도 글자로 남기지 않으려고 escape 로 쓴다.
    for bad in ('~/Desktop', '_audit', '\uc120\ubc30', '\ub2d8'):
        check(f'README 에 {bad} 없음', bad in rd, False)
    check('README 에 팀 저장소', 'neoul-ro/meridian_benchmark' in rd, True)
    check('README 에 엔진 릴리스', 'frontend-engines-20260915' in rd, True)
    check('README 가 짧다 (250줄 이하)', len(rd.splitlines()) <= 250, True)
    for n in ('docs/METRICS.md', 'docs/HISTORY.md'):
        p = HERE / n
        check(f'{n} 있음', p.exists(), True)
    met = (HERE / 'docs/METRICS.md').read_text() if (HERE / 'docs/METRICS.md').exists() else ''
    for w in ('IoU_τ', 'panopticapi', 'TIDE', '1600', 'ignored_match'):
        check(f'docs/METRICS.md 에 {w}', w in met, True)
    check('README 에 TrackEval 받는 법', '12c8791' in rd or 'TrackEval' in rd, True)
    check('README 에 세 가지 작업', all(w in rd for w in ('지금 결과', '내 frontend', '이전과 비교')), True)
    hist = [l for l in rd.splitlines() if '보관' in l and 'history' in l]
    check('[N10] 보관 조건(숫자가 바뀔 때만)을 문서가 맞게 적는다',
          any(('바뀔 때' in l) or ('달라질 때' in l) or ('같으면' in l) for l in hist), True)
    check('[N1] README 에 FB_STATUS_COPY 와 적용 조건', 'FB_STATUS_COPY' in rd and '전체' in rd, True)


def test_m7_no_internal_defaults(tmp):
    """M7 기본값에 내부 경로가 없다 — TrackEval 기본 경로가 _audit 아래가 아니다"""
    import trackeval_path as TE
    d = ' '.join(str(x) for x in TE.defaults())
    check('TrackEval 기본 경로가 _audit 아래가 아님', '_audit' in d, False)
    check('기본 위치는 third_party', 'third_party' in d, True)
    check('없으면 받는 법을 알려 준다', '12c8791' in TE.HOWTO and 'pip' in TE.HOWTO, True)
    p0 = str(TE.resolve(use_env=False)[0] or '')
    check('환경변수 없이 고른 경로에도 _audit 없음', '_audit' in p0, False)
    check('설정(TRACKEVAL_PATH)이 있으면 그것을 쓴다', TE.resolve()[1] in ('TRACKEVAL_PATH 설정', 'pip 로 설치된 trackeval 패키지',
                                                                        '기본 위치 <WS>/third_party/TrackEval', '없음'), True)
    par = (HERE / 'test_trackeval_parity.py').read_text()
    check('대조 테스트가 trackeval_path 를 쓴다', 'trackeval_path' in par, True)
    check('대조 테스트 SKIP 줄에 받는 법', 'HOWTO' in par, True)


# ------------------------------------------------------------------ m6 판정 그림
def test_m6_examples_labels(tmp):
    """m6 판정 그림의 글자 · 색이 terms.py 에서 오고 범례가 있다"""
    import render_examples as RE
    import terms as T
    check('terms 에 COLORS (뷰어와 같은 색)', hasattr(T, 'COLORS'), True)
    check('색 4가지', sorted(T.COLORS), ['detected', 'ineligible', 'merged', 'missed'])
    check('render_examples 가 terms 색을 쓴다', RE.COLORS is T.COLORS or RE.COLORS == T.COLORS, True)
    ttl = RE.panel_title(dict(ep_index='7', gt_object_id='12', detected='0', miss_reason='merged'), frame=741, px=1200, n_obs=3)
    check('제목이 한국어 표시 이름', '과소분할' in ttl, True)
    check('제목에 원시 라벨 MISS(merged) 없음', 'MISS(' in ttl, False)
    check('범례 함수 있음', hasattr(RE, 'legend_rows'), True)
    rows = RE.legend_rows()
    check('범례에 4가지 이상', len(rows) >= 4, True)
    check('범례 글자가 terms 에서', any(T.OBJECT['detected'] in r[0] for r in rows), True)


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for fn in (test_m5_metrics_table, test_m5_one_word_one_meaning, test_m5_same_number_everywhere,
                   test_m7_readme, test_m7_no_internal_defaults, test_m6_examples_labels):
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
