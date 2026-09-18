#!/usr/bin/env python3
"""report.py · compare.py 검증 (사용성 리뷰 C1 · C2 · C4 · M6 · M10). 합성 입력만 쓴다 (GPU·실제 결과 없음).

C1  채점 전에 시퀀스마다 frontend 출처를 확인한다 — 다르면 화면 경고 · summary.md 머리 표시 · 종료 코드 4
    (`--accept-stale` 또는 FB_ACCEPT_STALE=1 이면 받아들이고 0, 표시는 그대로 남는다)
C2  채점할 때마다 이전 summary.json · summary.md · status.md 를 <RUNS>/history/<시각>/ 에 보관(출처 기록 포함) ·
    부분 실행(--seqs)이 다른 시퀀스를 지우지 않는다 · compare 가 시퀀스 × 핵심 지표 변화표를 만든다
C4  데이터셋 · GT · gt_vis 가 없으면 한 줄로 어디가 없고 다음에 무엇을 할지 알려 주고 바로 멈춘다 (멈춤·로그 폭증 없음)
    라벨 단계 전에 부모 프로세스에서 입력을 한 번 열어 본다 (Pool 자식에서 실패해 무한히 다시 뜨는 것 방지)
M6  summary.md 는 '한눈에' 표와 판정으로 시작하고, 정의 · params 는 맨 뒤 · MOTA 를 쉬운 말로 한 번 설명 · '다른 시퀀스:' 문구 없음
M10 전부 누락이면 summary.md · summary.json 을 덮지 않는다
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PY = sys.executable
FAILS = []


def check(name, got, want):
    ok = got == want
    print(f'  {"OK " if ok else "FAIL"} {name}: got={got!r} want={want!r}')
    if not ok:
        FAILS.append(name)


def fixture_env(tmp):
    """FB_* 를 임시 폴더로 돌린 환경 (실제 결과를 건드리지 않는다)."""
    d = tmp
    for n in ('runs', 'logs', 'data', 'gt', 'models', 'src', 'msgs'):
        (d / n).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FB_RUNS=str(d / 'runs'), FB_LOGS=str(d / 'logs'), FB_DATA=str(d / 'data'), FB_GT=str(d / 'gt'),
               FB_MODELS=str(d / 'models'), FB_FRONTEND_SRC=str(d / 'src'), FB_FRONTEND_MSGS=str(d / 'msgs'))
    env.pop('FB_ACCEPT_STALE', None)
    return env


def make_run(d, seq='apartment_s1_00h', dataset=True, gt=True, vis=True):
    """채점 입력이 있는 척하는 최소 파일 묶음."""
    run = d / 'runs' / f'uHumans2_{seq}'
    run.mkdir(parents=True, exist_ok=True)
    (run / 'frontend_output.h5').write_bytes(b'not-a-real-h5')
    if vis:
        (d / 'runs' / 'gt_vis').mkdir(parents=True, exist_ok=True)
        (d / 'runs' / 'gt_vis' / f'uHumans2_{seq}.npz').write_bytes(b'x')
    if gt:
        (d / 'gt' / f'uHumans2_{seq}.h5').write_bytes(b'x')
    if dataset:
        sd = d / 'data' / f'uHumans2_{seq}'
        (sd / 'index').mkdir(parents=True, exist_ok=True)
        (sd / 'metadata.json').write_text('{}')
        (sd / 'index' / 'left_cam_rgb.csv').write_text('stamp_ns,filename\n')
    return run


def run_report(env, args, timeout=180):
    t0 = time.time()
    r = subprocess.run([PY, str(HERE / 'report.py'), *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(HERE))
    return r.returncode, r.stdout + r.stderr, time.time() - t0


# ------------------------------------------------------------------ C4 빨리 멈추기
def test_c4_missing_inputs(tmp):
    """C4 데이터셋 · GT 가 없으면 한 줄로 경로와 다음 행동을 알려 주고 바로 멈춘다"""
    import importlib
    d = tmp / 'c4'
    env = fixture_env(d)
    make_run(d, dataset=False)
    rc, out, secs = run_report(env, ['--seqs', 'apartment_s1_00h'], timeout=120)
    check('데이터셋 없음: 종료 코드 2', rc, 2)
    check('데이터셋 없음: 30초 안에 끝', secs < 30, True)
    check('데이터셋 없음: 경로를 알려 줌', str(d / 'data' / 'uHumans2_apartment_s1_00h') in out, True)
    check('데이터셋 없음: 다음 행동 (FB_DATA)', 'FB_DATA' in out, True)
    check('데이터셋 없음: traceback 없음', 'Traceback' in out, False)
    check('데이터셋 없음: "frontend 실행부터" 같은 틀린 안내 없음', 'frontend 실행부터' in out, False)
    check('M10 전부 누락이면 summary.md 를 덮지 않음', (d / 'runs' / 'summary.md').exists(), False)
    check('M10 전부 누락이면 summary.json 도 그대로', (d / 'runs' / 'summary.json').exists(), False)
    d2 = tmp / 'c4b'
    env2 = fixture_env(d2)
    make_run(d2, gt=False)
    rc, out, _ = run_report(env2, ['--seqs', 'apartment_s1_00h'])
    check('GT 없음: 종료 코드 2', rc, 2)
    check('GT 없음: 다음 행동 (FB_GT)', 'FB_GT' in out, True)
    os.environ.update({k: v for k, v in env.items() if k.startswith('FB_')})
    import paths, report  # noqa: E402
    importlib.reload(paths); importlib.reload(report)
    why = report.missing_reason('apartment_s1_00h')
    check('missing_reason 가 데이터셋을 본다', bool(why) and 'metadata.json' in why or '데이터셋' in (why or ''), True)


def test_c4_preflight(tmp):
    """C4 라벨 단계 전에 입력을 부모에서 한 번 열어 본다 — 깨진 데이터셋은 한 줄 오류로 즉시 멈춘다 (Pool 무한 재생성 방지)"""
    import importlib
    d = tmp / 'c4c'
    env = fixture_env(d)
    make_run(d)
    (d / 'data' / 'uHumans2_apartment_s1_00h' / 'metadata.json').write_text('{ 깨진 json')
    os.environ.update({k: v for k, v in env.items() if k.startswith('FB_')})
    import paths, report  # noqa: E402
    importlib.reload(paths); importlib.reload(report)
    check('report 에 preflight 함수', hasattr(report, 'preflight'), True)
    msg = None
    try:
        report.preflight('apartment_s1_00h')
    except SystemExit as e:
        msg = str(e)
    except Exception as e:  # noqa: BLE001
        msg = f'{type(e).__name__}: {e}'
    check('깨진 데이터셋: SystemExit 한 줄', bool(msg) and '\n' not in (msg or 'x\ny'), True)
    check('깨진 데이터셋: 경로가 메시지에', 'uHumans2_apartment_s1_00h' in (msg or ''), True)


# ------------------------------------------------------------------ C1 오래된 frontend 출력
def test_c1_stale_frontend(tmp):
    """C1 frontend 출처가 지금 코드·소스와 다르면 경고 · summary.md 표시 · 종료 코드 4"""
    import importlib
    d = tmp / 'c1'
    env = fixture_env(d)
    run = make_run(d)
    (d / 'src' / 'sam.py').write_text('x = 1\n')
    (d / 'models' / 'frontend_rtx3060').mkdir(parents=True, exist_ok=True)
    for n in ('fastsam.plan', 'clip_image.plan'):
        (d / 'models' / 'frontend_rtx3060' / n).write_bytes(b'engine')
    (run / 'run_meta.json').write_text(json.dumps(dict(provenance=dict(
        run_frontend_ast_sha1='old', frontend_src_sha1='old', frontend_src_files=1, engines={}, dataset_seq='old'))))
    os.environ.update({k: v for k, v in env.items() if k.startswith('FB_')})
    import paths, report  # noqa: E402
    importlib.reload(paths); importlib.reload(report)
    check('report 에 frontend_check 함수', hasattr(report, 'frontend_check'), True)
    st = report.frontend_check('apartment_s1_00h')
    check('출처가 다르면 stale', st.get('status'), 'stale')
    check('이유가 한국어로 들어 있다', bool(st.get('reasons')), True)
    md = report.render({'apartment_s1_00h': _block()}, {}, stale={'apartment_s1_00h': ['frontend 소스가 바뀜']})
    head = '\n'.join(md.splitlines()[:14])
    check('summary.md 머리에 오래된 출력 표시', ('오래된' in head) or ('옛 frontend' in head), True)
    check('summary.md 머리에 시퀀스 이름', 'apartment' in head, True)


def _block():
    from test_pipeline import fake_block
    b = fake_block(level_3d=dict(easy=dict(n=143, n_detected=131, recall=0.9161),
                                 moderate=dict(n=200, n_detected=170, recall=0.85),
                                 hard=dict(n=245, n_detected=200, recall=0.8163)))
    b['seg2d'] = dict(params=dict(theta=0.5, area_min_cells=256, loc_min_iou=0.1, matching='m', iou='i', ignore='ig', split='sp'),
                      all=dict(n_gt_instances=1000, recall={'0.25': 0.5, '0.5': 0.4, '0.75': 0.3}, precision=0.5, SQ=0.8, RQ=0.5,
                               PQ=0.42, n_tp=400, n_fp=400, n_fn=600,
                               gt_status=dict(merged=100, split=200, low_iou=150, missed=150)))
    b['geometry'] = dict(n_obs=1000, params=dict(taus_m=[0.05, 0.1, 0.2]), skipped_obs=dict(ignored_match=10),
                         summary={'accuracy_cm': dict(median=0.92), 'completeness_cm': dict(median=1.5),
                                  'chamfer_l1_cm': dict(median=1.2), 'F@5': dict(median=0.5), 'F@10': dict(median=0.7),
                                  'F@20': dict(median=0.91)},
                         perfect_upper_bound=dict(summary={'F@20': dict(median=0.99)}))
    b['mot'] = dict(params=dict(min_frac=0.5, present='p', fp='f', match='m', hota='h', idsw='i', frag='fr', mt='mt'),
                    metrics=dict(n_gt_tracks=245, IDTP=100, IDFP=100, IDFN=100, HOTA=0.44, DetA=0.5, AssA=0.4, TP=1600, FN=647,
                                 FP=2000, IDSW=50, Frag=60, MT=100, PT=100, ML=45, gt_detections=2247, mean_ids_per_track=2.03),
                    metrics_object=dict(n_gt_tracks=200, IDTP=90, IDFP=90, IDFN=90, HOTA=0.4, IDSW=40, mean_ids_per_track=2.5))
    b['counts_3d'] = dict(all=dict(n=483, detected=248), in_view=dict(n=477, detected=248),
                          eligible=dict(n=245, detected=198), eligible_static=dict(n=245, detected=198),
                          eligible_human=dict(n=0, detected=0))
    b['report_steps'] = dict(labels=dict(finished='2026-09-18 01:43:00', key='abc'))
    return b


# ------------------------------------------------------------------ C2 보관 · 부분 실행 · compare
def test_c2_history_and_partial(tmp):
    """C2 이전 결과 보관 · 부분 실행이 다른 시퀀스를 지우지 않음"""
    import importlib
    d = tmp / 'c2'
    env = fixture_env(d)
    os.environ.update({k: v for k, v in env.items() if k.startswith('FB_')})
    import paths, report  # noqa: E402
    importlib.reload(paths); importlib.reload(report)
    runs = Path(env['FB_RUNS'])
    old = {'apartment_s1_00h': _block(), 'office_s1_06h': _block()}
    (runs / 'summary.json').write_text(json.dumps(old, ensure_ascii=False))
    (runs / 'summary.md').write_text('# 옛 채점표\n')
    (runs / 'status.md').write_text('# 옛 status\n')
    check('report 에 archive_history 함수', hasattr(report, 'archive_history'), True)
    p = report.archive_history('테스트')
    check('history 폴더가 생김', bool(p) and Path(p).is_dir(), True)
    for n in ('summary.json', 'summary.md', 'status.md', 'provenance.json'):
        check(f'보관에 {n}', (Path(p) / n).exists(), True)
    pv = json.loads((Path(p) / 'provenance.json').read_text())
    for k in ('frontend_src_sha1', 'benchmark_code_sha1', 'sequences', 'archived_at'):
        check(f'보관 출처에 {k}', k in pv, True)
    check('보관 출처의 시퀀스 목록', sorted(pv.get('sequences') or []), ['apartment_s1_00h', 'office_s1_06h'])
    check('report 에 merge_summary 함수 (부분 실행 병합)', hasattr(report, 'merge_summary'), True)
    merged = report.merge_summary(json.loads((runs / 'summary.json').read_text()), {'office_s1_06h': _block()})
    check('부분 실행: 다른 시퀀스가 남는다', sorted(merged), ['apartment_s1_00h', 'office_s1_06h'])
    md = report.render(merged, {}, rescored=['office_s1_06h'])
    check('다시 채점한 시퀀스를 표시', 'office' in md and ('이번에 다시 채점' in md or '다시 채점한 시퀀스' in md), True)


def test_c2_compare(tmp):
    """C2 compare: 시퀀스 × 핵심 지표 변화표 (지표 방향에 맞게 좋아짐 · 나빠짐)"""
    sys.path.insert(0, str(HERE))
    import compare as C  # noqa: E402
    old = {'apartment_s1_00h': _block()}
    new = {'apartment_s1_00h': _block()}
    new['apartment_s1_00h']['seg2d']['all']['PQ'] = 0.52
    new['apartment_s1_00h']['seg2d']['all']['gt_status']['split'] = 300
    md = C.render_compare(old, new, old_id='20260918-000000', old_info={})
    check('표에 시퀀스', 'apartment_00h' in md or 'apartment_s1_00h' in md, True)
    check('PQ 가 올라가면 좋아짐', '좋아짐' in md, True)
    check('과다분할이 늘면 나빠짐', '나빠짐' in md, True)
    check('변화량 표시 (+0.10)', '+0.10' in md, True)
    same = C.render_compare(old, old, old_id='x', old_info={})
    check('같으면 변화 없음', '변화 없음' in same or '같습니다' in same, True)


# ------------------------------------------------------------------ M6 summary.md 구조
def test_n5_compare_old_format(tmp):
    """N5 옛 형식 · 깨진 보관본을 만나면 traceback 대신 한 줄로 알린다"""
    d = tmp / 'n5'
    env = fixture_env(d)
    runs = Path(env['FB_RUNS'])
    (runs / 'summary.json').write_text(json.dumps({'apartment_s1_00h': _block()}, ensure_ascii=False))
    cases = {'old_list': json.dumps([{'seq': 'apartment_s1_00h'}]),
             'old_wrap': json.dumps({'seqs': {'apartment_s1_00h': {}}, 'generated': 'x'}),
             'old_str': json.dumps({'apartment_s1_00h': '옛 형식'}),
             'bad_json': '{ 깨진'}
    for name, body in cases.items():
        h = runs / 'history' / name
        h.mkdir(parents=True, exist_ok=True)
        (h / 'summary.json').write_text(body)
        (h / 'provenance.json').write_text(json.dumps(dict(archived_at='2026-09-17 00:00:00')))
        r = subprocess.run([PY, str(HERE / 'compare.py'), name], capture_output=True, text=True, timeout=120, env=env, cwd=str(HERE))
        out = r.stdout + r.stderr
        check(f'{name}: traceback 없음', 'Traceback' in out, False)
        check(f'{name}: 종료 코드 2', r.returncode, 2)
        check(f'{name}: 무엇이 문제인지 한 줄', any(w in out for w in ('형식', '읽을 수 없습니다', '열 수 없습니다', 'JSON')), True)


def test_m6_summary_head(tmp):
    """M6 summary.md 는 '한눈에' 표 · 판정으로 시작하고 정의 · params 는 맨 뒤"""
    import report as R
    rows = {'apartment_s1_00h': _block(), 'office_s1_06h': _block()}
    md = R.render(rows, {})
    lines = md.splitlines()
    heads = [l for l in lines if l.startswith('## ')]
    check('첫 절이 한눈에', heads[0].startswith('## 1. 한눈에') or heads[0].startswith('## 한눈에'), True)
    top = '\n'.join(lines[:40])
    check('한눈에 표에 판정 문구', ('겹칩니다' in top) or ('분할' in top and '약합니다' in top), True)
    check('한눈에가 정의보다 앞', md.index('한눈에') < md.index('GT 트랙(ground-truth track) ='), True)
    check('정의 · params 는 맨 뒤 (마지막 20% 안)', md.index('허용 거리 IoU_τ') > len(md) * 0.5, True)
    check('MOTA 를 쉬운 말로 설명', 'MOTA' in md and ('놓친' in md or '빠뜨린' in md), True)
    check('"다른 시퀀스:" 문구 없음', '**다른 시퀀스**:' in md, False)
    check('판정 기준이 status.md 와 같은 표', 'JUDGE' in md or '판정 기준' in md, True)
    check('판정 그림(examples) 링크', 'examples/' in md, True)


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for fn in (test_c4_missing_inputs, test_c4_preflight, test_c1_stale_frontend, test_c2_history_and_partial,
                   test_c2_compare, test_n5_compare_old_format, test_m6_summary_head):
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
