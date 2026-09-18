#!/usr/bin/env python3
"""status_md.py 검증 — 만든 status.md 를 다시 읽어 표의 모든 칸과 글머리표 숫자를 summary.json · tests_summary.json 과 대조.

검증기(verify)는 status_md.py 의 포맷 함수를 쓰지 않고 따로 계산한다 (같은 코드로 같은 답을 내는 동어반복 방지).
  표 열     시퀀스 | 트랙 재현율 (Easy) | F@20cm | PQ | IDF1 | HOTA_α
  트랙 재현율 (Easy)  level_3d.easy 의 n_detected / n → % 소수 1자리 (반올림 half-up, 개수에서 한 번만 — terms.METRICS)
  F@20cm   geometry.summary['F@20'].median → 소수 2자리
  PQ       seg2d.all.PQ → 소수 2자리
  IDF1     mot.metrics 의 IDTP·IDFP·IDFN 로 2·IDTP/(2·IDTP+IDFP+IDFN) → 소수 2자리 (개수 없으면 IDF1 값)
  HOTA_α   mot.metrics.HOTA → 소수 2자리
  글머리표  '- ' 로 시작하는 줄의 숫자는 전부 JSON 에서 나온 값이어야 한다
  용어      "3D 재현율" 없음 · F@20cm 를 '20cm 이내 비율'(정밀도 한쪽)로 설명하지 않음 · ID 수는 'GT 트랙당 예측 ID'
  자체 검증 수  tests_summary.json totals.n_ok · 통과 파일 수
  모양 [수정 ⑤ F3 — 사용자 요청 "너무 난잡해 핵심만"]
            제목 · 둘째 줄 '<날짜> · <작성자>'(FB_AUTHOR, 기본 김주영) · 생성 정보는 HTML 주석 · 소개 2줄 · 결과 표 ·
            글머리표 4개 = **위치** · **분할** · **추적** 한 줄씩(숫자 + 짧은 판정 한 마디) + 용어 한 줄(트랙 재현율 · Easy · HOTA_α) ·
            뷰어 한 줄 · 한계 한 줄(검증 항목 수) · 끝 질문. 글머리표는 120자 이하 · 괄호 안에 괄호 없음 ·
            영어 용어 괄호는 문서 전체에서 같은 것이 한 번만 · 임시 문구 없음
  판정 문구  status_md.JUDGE 의 기준으로 고른다 (좋다고 말하려면 가장 나쁜 시퀀스도 기준을 넘어야 한다)
실행: python test_status_md.py                       합성 입력
      python test_status_md.py --verify STATUS SUMMARY TESTS   실제 파일 검증
"""
import json
import re
import sys
import traceback
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FAILS = []
COLS = ['시퀀스', '트랙 재현율 (Easy)', 'F@20cm', 'PQ', 'IDF1', 'HOTA_α']


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


def d2(x):
    return '–' if x is None else str(Decimal(repr(float(x))).quantize(Decimal('0.01'), ROUND_HALF_UP))


def pct1(k, n):
    """percent 지표 형식 = 소수 1자리 (terms.METRICS). 정수 연산 half-up: floor(1000k/n + 1/2)."""
    if not n:
        return '–'
    x = (2000 * k + n) // (2 * n)
    return f'{x // 10}.{x % 10}%'


def short(s):
    return s.replace('_s1_', '_')


def expected_cells(r):
    lv = (r.get('level_3d') or {}).get('easy') or {}
    geo = ((r.get('geometry') or {}).get('summary') or {}).get('F@20') or {}
    seg = (r.get('seg2d') or {}).get('all') or {}
    m = (r.get('mot') or {}).get('metrics') or {}
    if all(k in m for k in ('IDTP', 'IDFP', 'IDFN')) and (2 * m['IDTP'] + m['IDFP'] + m['IDFN']):
        p_, q_ = 2 * m['IDTP'], 2 * m['IDTP'] + m['IDFP'] + m['IDFN']
        x = (200 * p_ + q_) // (2 * q_)                               # 정수 연산 half-up (소수 2자리)
        idf1 = f'{x // 100}.{x % 100:02d}'
    else:
        idf1 = d2(m.get('IDF1'))
    return [pct1(lv.get('n_detected', 0), lv.get('n', 0)), d2(geo.get('median')), d2(seg.get('PQ')), idf1, d2(m.get('HOTA'))]


def allowed_numbers(summary, tests):
    """글머리표에 나와도 되는 숫자 문자열 (JSON 에서 계산)."""
    ok = set()
    add = lambda v: ok.add(str(v))
    vals = {'f20': [], 'pq': [], 'ids': [], 'rec': []}
    for s, r in summary.items():
        c = expected_cells(r)
        vals['rec'].append(c[0].rstrip('%')); vals['f20'].append(c[1]); vals['pq'].append(c[2])
        m = (r.get('mot') or {}).get('metrics') or {}
        if m.get('mean_ids_per_track') is not None:
            vals['ids'].append(str(Decimal(repr(m['mean_ids_per_track'])).quantize(Decimal('0.1'), ROUND_HALF_UP)))
        st = ((r.get('seg2d') or {}).get('all') or {}).get('gt_status') or {}
        for k in ('split', 'merged', 'low_iou', 'missed', 'tp'):
            add(st.get(k, 0))
        add(c[3]); add(c[4])
        add((r.get('visibility') or {}).get('area_min_px'))
        crit = r.get('level_3d_criteria') or {}
        for lv in crit.values():
            for k, v in lv.items():
                add(v); add(int(round(v * 100)) if isinstance(v, float) and v <= 1 else v)
        mp = (r.get('mot') or {}).get('params') or {}
        for k in ('min_frac', 'iou_threshold'):
            if k in mp:
                add(mp[k])
        sp = r.get('params') or {}
        if sp.get('tau_m') is not None:
            add(int(round(sp['tau_m'] * 100)))
        for k in ('iou_threshold', 'loc_min_iou'):
            if sp.get(k) is not None:
                add(sp[k])
        gp = (r.get('geometry') or {}).get('params') or {}
        for t in gp.get('taus_m') or []:
            add(int(round(t * 100)))
    for k, v in vals.items():
        for x in v:
            add(x)
    for x in (sum(int(((r.get('seg2d') or {}).get('all') or {}).get('gt_status', {}).get(k, 0)) for r in summary.values())
              for k in ('split', 'merged', 'low_iou', 'missed')):
        add(x)
    t = (tests or {}).get('totals') or {}
    for k in ('n_ok', 'files', 'passed', 'failed', 'skipped', 'n_fail'):
        if k in t:
            add(t[k])
    add(len(summary))
    return ok


def verify(md, summary, tests):
    """→ 틀린 곳 목록 (빈 목록이면 통과)."""
    errs = []
    lines = md.splitlines()
    hi = next((i for i, l in enumerate(lines) if l.startswith('| 시퀀스')), None)
    if hi is None:
        return ['표 머리 없음']
    head = [c.strip() for c in lines[hi].strip('|').split('|')]
    if head != COLS:
        errs.append(f'열 {head} ≠ {COLS}')
    seen = set()
    for l in lines[hi + 2:]:
        if not l.startswith('|'):
            break
        cells = [c.strip() for c in l.strip('|').split('|')]
        name = cells[0]
        full = next((s for s in summary if short(s) == name or s == name), None)
        if full is None:
            errs.append(f'summary.json 에 없는 시퀀스 {name}'); continue
        seen.add(full)
        exp = expected_cells(summary[full])
        for col, got, want in zip(COLS[1:], cells[1:], exp):
            if got != want:
                errs.append(f'{name} · {col}: {got} ≠ {want}')
    for s in summary:
        if s not in seen:
            errs.append(f'표에 빠진 시퀀스 {s}')
    allowed = allowed_numbers(summary, tests)
    bullets = [l for l in lines if l.startswith('- ')]
    for b in bullets:
        text = re.sub(r'F@\d+cm|P@\d+cm|R@\d+cm|HOTA_α|DetA_α|AssA_α|IDF1|@IoU0\.\d+|uHumans2|_\d\dh|2D|3D|\bs1\b|1이 만점', '', b)   # '1이 만점' = 척도 표기
        for num in re.findall(r'\d+(?:\.\d+)?', text):
            if num not in allowed:
                errs.append(f'글머리표 숫자 {num} 가 JSON 에 없음: {b[:60]}')
    if '3D 재현율' in md:
        errs.append('"3D 재현율" 표현 (감사 D 충돌)')
    if any('F@20cm' in b and ('이내 비율' in b or '정밀도' in b) for b in bullets):
        errs.append('F@20cm 를 20cm 이내 비율(정밀도 한쪽)로 설명함')
    errs += style_errors(md)
    if 'GT 트랙당 예측 ID' not in md or '물체당' in md:
        errs.append('ID 수가 GT 트랙당 예측 ID 가 아님')
    if len(bullets) != 4:
        errs.append(f'글머리표 {len(bullets)}개 (위치 · 분할 · 추적 · 용어 4개)')
    for w in ('바꾸는 중', '바뀌는 중', '바뀔 수 있습니다', '검토 중'):
        if w in md:
            errs.append(f'임시 문구 "{w}"')
    heads = [l for l in lines if l.startswith('## ')]
    if heads != ['## 결과', '## 뷰어', '## 한계']:
        errs.append(f'절 제목 {heads} ≠ 결과 · 뷰어 · 한계')
    body = [l for l in lines if l.strip()]
    closing = config_value('FB_CLOSING')
    if closing and (not body or body[-1].strip() != closing):
        errs.append(f'끝 질문 "{closing}" 없음')
    lim = lines[lines.index('## 한계') + 1:] if '## 한계' in lines else []
    lim = [l for l in lim if l.strip()]
    want_lim = 2 if config_value('FB_CLOSING') else 1
    if len(lim) != want_lim:
        errs.append(f'한계 절: 한 줄{" + 끝 질문" if want_lim == 2 else ""}이어야 함 ({len(lim)}줄)')
    view = lines[lines.index('## 뷰어') + 1:lines.index('## 한계')] if '## 뷰어' in lines and '## 한계' in lines else []
    if len([l for l in view if l.strip()]) != 1:
        errs.append('뷰어 절이 한 줄이 아님')
    t = (tests or {}).get('totals') or {}
    if t and f'{t.get("n_ok")}항목' not in md:
        errs.append(f'자체 검증 수 {t.get("n_ok")}항목 없음')
    return errs


MAX_BULLET = 120


def config_value(name, default=''):
    """m5 설정 확인용 — 환경변수 → local.env → 기본값 (status_md.py 를 쓰지 않고 따로 읽는다)."""
    import os
    v = os.environ.get(name)
    if v is None:
        for line in (HERE / 'local.env').read_text().splitlines() if (HERE / 'local.env').exists() else []:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line and line.split('=', 1)[0].strip() == name:
                v = line.split('=', 1)[1].strip()
                break
    return default if v is None else v


def style_errors(md, author=None):
    """F3 짧은 형식 검사 → 오류 목록."""
    errs = []
    lines = md.splitlines()
    body = [l for l in lines if l.strip() and not l.startswith('>')]
    author = author if author is not None else config_value('FB_AUTHOR')
    want2 = r'(\d{4}-\d\d-\d\d|날짜 없음)' + (' · ' + re.escape(author) if author else '')
    if len(body) < 2 or not re.fullmatch(want2, body[1]):
        errs.append(f'둘째 줄이 "<날짜> · {author}" 가 아님: {body[1] if len(body) > 1 else ""!r}')
    if not re.search(r'<!--.*status_md\.py.*-->', md, re.S):
        errs.append('생성 정보 HTML 주석 없음')
    if any('status_md.py' in l for l in lines if not l.lstrip().startswith('<!--')):
        errs.append('생성 정보가 주석 밖에 보임')
    i0 = next((i for i, l in enumerate(body) if l.startswith('<!--')), None)
    if i0 is not None:
        intro = []
        for l in body[i0 + 1:]:
            if l.startswith('## '):
                break
            intro.append(l)
        if len(intro) != 2:
            errs.append(f'소개 {len(intro)}줄 (2줄이어야 함)')
    bullets = [l for l in lines if l.startswith('- ')]
    labels = [(re.match(r'- \*\*(.+?)\*\*:', b) or [None, None])[1] for b in bullets]
    if labels[:3] != ['위치', '분할', '추적']:
        errs.append(f'글머리표 이름 {labels[:3]} ≠ 위치 · 분할 · 추적')
    if len(bullets) >= 4 and not all(w in bullets[3] for w in ('트랙 재현율', 'Easy', 'HOTA_α')):
        errs.append('마지막 글머리표가 용어 한 줄(트랙 재현율 · Easy · HOTA_α)이 아님')
    for b in bullets:
        if len(b) > MAX_BULLET:
            errs.append(f'글머리표 {len(b)}자 > {MAX_BULLET}: {b[:40]}')
        depth = 0
        for ch in b:
            depth += ch == '('
            if depth > 1:
                errs.append(f'괄호 안에 괄호: {b[:40]}'); break
            depth -= ch == ')'
    eng = re.findall(r'\(([^()]*[A-Za-z][^()]*)\)', md)
    dup = sorted({t for t in eng if eng.count(t) > 1})
    if dup:
        errs.append(f'영어 괄호 용어가 두 번 이상: {dup}')
    return errs


def fake_summary():
    def seq(k, n, f20, pq, idtp, idfp, idfn, hota, ids, split, merged):
        return dict(
            params=dict(tau_m=0.2, iou_threshold=0.5, loc_min_iou=0.25),
            visibility=dict(area_min_px=1600),
            level_3d=dict(easy=dict(n=n, n_detected=k, recall=round(k / n, 4)), moderate=dict(n=n, n_detected=k), hard=dict(n=n, n_detected=k)),
            level_3d_criteria=dict(easy=dict(min_dim_gt=40, trunc_le=0.15, occ_le=0.2), moderate=dict(min_dim_gt=25, trunc_le=0.3, occ_le=0.5),
                                   hard=dict(min_dim_gt=25, trunc_le=0.5)),
            geometry=dict(summary={'F@20': dict(median=f20, mean=f20)}, params=dict(taus_m=[0.05, 0.1, 0.2])),
            seg2d=dict(all=dict(PQ=pq, gt_status=dict(tp=10, split=split, merged=merged, low_iou=3, missed=4))),
            mot=dict(metrics=dict(IDTP=idtp, IDFP=idfp, IDFN=idfn, IDF1=round(2 * idtp / (2 * idtp + idfp + idfn), 4), HOTA=hota,
                                  mean_ids_per_track=ids), params=dict(min_frac=0.5)))
    return {'apartment_s1_00h': seq(168, 204, 0.8455, 0.4235, 1167, 1951, 1080, 0.4813, 2.5612, 71, 233),
            'office_s1_06h': seq(301, 316, 0.8295, 0.3391, 2000, 3000, 2500, 0.4306, 3.1049, 373, 96)}


def fake_tests():
    return dict(totals=dict(files=10, passed=10, failed=0, skipped=0, n_ok=512, n_fail=0))


def test_render_and_verify():
    """합성 summary → status.md → 칸별 대조"""
    import status_md as SMD
    summ, tests = fake_summary(), fake_tests()
    md = SMD.render_status(summ, tests)
    print('\n'.join('     ' + l for l in md.splitlines()[:14]))
    errs = verify(md, summ, tests)
    for e in errs:
        print('     ·', e)
    check('검증기 오류 0', len(errs), 0)
    check('82.35…% 는 82.4% (개수에서 한 번 반올림)', '| 82.4% |' in md, True)
    bad = md.replace('| 82.4% |', '| 82.3% |', 1)
    check('검증기가 틀린 칸을 잡는다 (바꾼 칸 → 오류)', len(verify(bad, summ, tests)) > 0, True)
    bad2 = md.replace('512항목', '513항목')
    check('검증기가 자체 검증 수 불일치를 잡는다', len(verify(bad2, summ, tests)) > 0, True)
    body = [l for l in md.splitlines() if l.strip()]
    who = config_value('FB_AUTHOR')
    check(f'[F3] 둘째 줄 = "<날짜> · {who}"', bool(re.fullmatch(r'(\d{4}-\d\d-\d\d|날짜 없음)' + (' · ' + re.escape(who) if who else ''), body[1])), True)
    check('[F3] 생성 정보는 HTML 주석 안', ('<!--' in md) and not any('status_md.py' in l for l in md.splitlines() if not l.startswith('<!--')), True)
    bullets = [l for l in md.splitlines() if l.startswith('- ')]
    check('[F3] 글머리표 4개 · 모두 120자 이하', (len(bullets), max(len(b) for b in bullets) <= MAX_BULLET), (4, True))
    check('[F3] 형식 검사 오류 0', style_errors(md), [])
    long = md.replace(bullets[0], bullets[0] + ' 덧붙인 설명' * 20)
    check('[F3] 검사기가 긴 글머리표를 잡는다', any('자 >' in e for e in style_errors(long)), True)
    nested = md.replace(bullets[1], bullets[1] + ' (바깥 (안쪽))')
    check('[F3] 검사기가 괄호 안 괄호를 잡는다', any('괄호 안에 괄호' in e for e in style_errors(nested)), True)


def test_judge_thresholds():
    """[F3] 판정 문구는 코드에 적힌 기준으로 고른다 (경계값 포함)"""
    import os
    import status_md as SMD
    J, JT = SMD.judge, SMD.JUDGE
    check('위치: F 최솟값 0.90 → 잘 · 0.89 → 대체로 · 0.75 → 대체로 · 0.74 → 어긋남', [J('position', x) for x in (0.90, 0.89, 0.75, 0.74)],
          [JT['position'][0][1], JT['position'][1][1], JT['position'][1][1], JT['position'][2][1]])
    check('분할: PQ 최솟값 0.70 → 좋음 · 0.69 → 보통 · 0.50 → 보통 · 0.49 → 약함', [J('segmentation', x) for x in (0.70, 0.69, 0.50, 0.49)],
          [JT['segmentation'][0][1], JT['segmentation'][1][1], JT['segmentation'][1][1], JT['segmentation'][2][1]])
    check('추적: ID 수 최댓값 1.2 → 거의 유지 · 1.3 → 가끔 · 1.5 → 가끔 · 1.6 → 자주', [J('tracking', x) for x in (1.2, 1.3, 1.5, 1.6)],
          [JT['tracking'][0][1], JT['tracking'][1][1], JT['tracking'][1][1], JT['tracking'][2][1]])
    summ = fake_summary()
    md = SMD.render_status(summ, fake_tests())
    check('합성 입력 (F 0.83~0.85 · PQ 0.34~0.42 · ID 2.6~3.1) 문구', [JT['position'][1][1] in md, JT['segmentation'][2][1] in md,
                                                                   JT['tracking'][2][1] in md], [True, True, True])
    old = os.environ.get('FB_AUTHOR')
    os.environ['FB_AUTHOR'] = '테스트작성자'
    try:
        md2 = SMD.render_status(summ, fake_tests())
    finally:
        if old is None:
            os.environ.pop('FB_AUTHOR', None)
        else:
            os.environ['FB_AUTHOR'] = old
    check('FB_AUTHOR 로 작성자를 바꾼다', [l for l in md2.splitlines() if l.strip()][1].endswith(' · 테스트작성자'), True)


def test_m5_options():
    """m5 작성자 · 끝 질문 · 뷰어 경로가 하드코딩이 아니다 (옵션 → 환경변수 → local.env)"""
    import os
    import status_md as SMD
    summ, tests = fake_summary(), fake_tests()
    md = SMD.render_status(summ, tests, author='홍길동', closing='검토 부탁드립니다.')
    body = [l for l in md.splitlines() if l.strip()]
    check('--author 옵션이 이긴다', body[1].endswith(' · 홍길동'), True)
    check('--closing 옵션이 이긴다', body[-1], '검토 부탁드립니다.')
    md2 = SMD.render_status(summ, tests, author='', closing='')
    body2 = [l for l in md2.splitlines() if l.strip()]
    check('작성자 없이도 날짜 줄', bool(re.fullmatch(r'\d{4}-\d\d-\d\d|날짜 없음', body2[1])), True)
    check('끝 질문을 끌 수 있다', body2[-1].endswith('?'), False)
    check('local.env 를 읽는다', SMD.local_env().get('FB_AUTHOR'), config_value('FB_AUTHOR'))
    rel = os.path.relpath(__import__('paths').RUNS, __import__('paths').WS)
    check('뷰어 경로가 실제 RUNS 기준', f'{rel}/viewer/index.html' in md, True)
    stale = json.loads(json.dumps(summ))
    k = list(stale)[0]
    stale[k].setdefault('frontend', {})['provenance_check'] = dict(status='stale', reasons=['frontend 소스가 바뀜'])
    md3 = SMD.render_status(stale, tests)
    check('C1 오래된 frontend 출력이면 머리에 경고', '⚠' in md3.splitlines()[6] or any('⚠' in l for l in md3.splitlines()[:10]), True)
    check('C1 경고에 시퀀스 이름', SMD.short(k) in md3, True)


def test_main_writes_runs(tmp_runs=None):
    """status_md.main 은 RUNS/status.md 에 쓴다 (Desktop 아님)"""
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        (runs / 'summary.json').write_text(json.dumps(fake_summary()))
        (runs / 'tests_summary.json').write_text(json.dumps(fake_tests()))
        dest = runs / 'copy' / 'status_copy.md'
        dest.parent.mkdir()
        r = subprocess.run([sys.executable, str(HERE / 'status_md.py')],
                           env=dict(os.environ, FB_RUNS=str(runs), FB_STATUS_COPY=str(dest)),
                           capture_output=True, text=True, timeout=120)
        out0 = r.stdout + r.stderr
        print('     ', (r.stdout + r.stderr).strip().splitlines()[-1:] if (r.stdout + r.stderr).strip() else '')
        check('종료 코드 0', r.returncode, 0)
        p = runs / 'status.md'
        check('RUNS/status.md 생성', p.exists(), True)
        if p.exists():
            check('생성 파일 검증 오류 0', verify(p.read_text(), fake_summary(), fake_tests()), [])
        check('[N1] 기본 결과 폴더가 아니면 사본을 만들지 않고 이유를 적는다', (dest.exists(), '사본 건너뜀' in out0), (False, True))
        r2 = subprocess.run([sys.executable, str(HERE / 'status_md.py'), '--copy-to', str(dest)],
                            env=dict(os.environ, FB_RUNS=str(runs)), capture_output=True, text=True, timeout=120)
        check('[m5] --copy-to 는 조건 없이 사본 생성', dest.exists() and dest.read_text() == p.read_text(), True)


def main():
    if '--verify' in sys.argv:
        i = sys.argv.index('--verify')
        md, summ, tests = (Path(x) for x in sys.argv[i + 1:i + 4])
        errs = verify(md.read_text(), json.loads(summ.read_text()), json.loads(tests.read_text()) if tests.exists() else {})
        for e in errs:
            print('  FAIL', e)
        print(f'  {"OK " if not errs else "FAIL"} {md} 칸·숫자 대조 (오류 {len(errs)})')
        sys.exit(1 if errs else 0)
    for fn in (test_render_and_verify, test_judge_thresholds, test_m5_options, test_main_writes_runs):
        print(f'[{fn.__name__}] {fn.__doc__.strip()}')
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=3)
            print(f'  FAIL {fn.__name__}: 예외 {type(e).__name__}: {e}')
            FAILS.append(fn.__name__)
    print('\n결과:', '전부 통과' if not FAILS else f'실패 {len(FAILS)}개 {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
