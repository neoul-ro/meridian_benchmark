#!/usr/bin/env python3
"""뷰어 JS 논리 검증 (node 로 실행, 브라우저 없이). 수정 ⑦ — M2 · M3 · m1 · m3.

viewer_template.html 의 <script id="fblib"> 묶음(순수 논리, DOM 없음)을 꺼내 node 로 돌린다.
  M2  pickAt 하나로 마우스 올림과 클릭이 같은 물체를 고른다 (정확한 칸 먼저 · 필터 반영).
  M3  missedGroups (원인별 미검출 목록) · kfPass (keyframe 2D/3D 상태 필터).
  m4  hatch (맵에서 채점 제외 물체를 빗금으로 — 색만으로 구분하지 않게).
  m3  parseHash · buildHash (#시퀀스/tab=…/kf=…/obj=… 왕복).
  m1  staleState (뷰어가 채점보다 오래됐는지).
  2차  imgScale (keyframe 그림을 자리에 맞춰 키움) · pickKf (고른 물체가 판정된 keyframe) · canReadSummary (summary.json 404 방지).
템플릿의 모든 <script> 구문도 node --check 로 본다 (따옴표가 깨지면 화면 전체가 빈다).
node 가 없으면 SKIP.
실행: python test_viewer_js.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent      # frontend_benchmark/ (테스트 파일은 tests/ 에 있다)
TESTS = Path(__file__).resolve().parent            # frontend_benchmark/tests/

JS_TEST = r'''
const L = require('./fblib.js');
let fails = 0;
function check(name, got, want) {
  const g = JSON.stringify(got), w = JSON.stringify(want), ok = g === w;
  if (!ok) fails++;
  console.log(`  ${ok ? 'OK ' : 'FAIL'} ${name}: got=${g} want=${w}`);
}

// ---------------- m3 URL 해시
check('해시 읽기', L.parseHash('#apartment_s1_00h/tab=kf/kf=239/obj=12'),
      { seq: 'apartment_s1_00h', tab: 'kf', kf: 239, obj: 12 });
check('& 로 이어 써도 읽음', L.parseHash('#tab=map&obj=7'), { tab: 'map', obj: 7 });
check('빈 해시', L.parseHash(''), {});
check('해시 쓰기', L.buildHash({ seq: 'office_s1_06h', tab: 'kf', kf: 239, obj: 12 }),
      '#office_s1_06h/tab=kf/obj=12/kf=239');
check('쓰고 읽으면 그대로', L.parseHash(L.buildHash({ seq: 's', tab: 'map', obj: 3, kf: 0 })),
      { seq: 's', tab: 'map', obj: 3, kf: 0 });
check('모르는 열쇠는 버린다', L.parseHash('#tab=kf/zzz=1'), { tab: 'kf' });

// ---------------- M2 선택 함수 하나
// 칸 → 물체: (10,10)=0 · (12,10)=1 · (10,12)=2 (물체 1 은 필터로 숨김)
const pick = new Map([[10 * 65536 + 10, 0], [12 * 65536 + 10, 1], [10 * 65536 + 12, 2]]);
const visAll = () => true, visNo1 = i => i !== 1;
check('정확한 칸이 먼저', L.pickAt(pick, 10, 10, visAll, 2), 0);
check('빈 칸이면 가까운 칸 (반경 안)', L.pickAt(pick, 11, 10, visAll, 2), 0);
check('반경 밖이면 없음', L.pickAt(pick, 40, 40, visAll, 2), -1);
check('필터로 숨긴 물체는 고르지 않는다 (반경 0)', L.pickAt(pick, 12, 10, visNo1, 0), -1);
check('숨긴 칸 위에서도 반경 안의 보이는 물체를 고른다', L.pickAt(pick, 12, 10, visNo1, 2), 0);
check('숨긴 칸 옆에서는 보이는 칸을 고른다', L.pickAt(pick, 11, 11, visNo1, 2), 0);
// 마우스 올림 = 클릭 (같은 인자 · 같은 함수)
let same = true;
for (let x = 6; x <= 16; x++) for (let y = 6; y <= 16; y++) {
  const hover = L.pickAt(pick, x, y, visNo1, 2), click = L.pickAt(pick, x, y, visNo1, 2);
  if (hover !== click) same = false;
}
check('올림과 클릭이 같은 결과', same, true);
check('반경 0 이면 정확한 칸만', L.pickAt(pick, 11, 10, visAll, 0), -1);

// ---------------- M3 미검출 목록 · keyframe 필터
const objs = [
  { id: 1, status: 'missed', rc: 'no_kf', kind: 'static' },
  { id: 2, status: 'missed', rc: 'no_kf', kind: 'static' },
  { id: 3, status: 'merged', rc: 'merged', kind: 'static' },
  { id: 4, status: 'detected', rc: '', kind: 'static' },
  { id: 5, status: 'missed', rc: 'low_iou', kind: 'human' },
];
const reasons = [{ code: 'merged', label: '합침' }, { code: 'low_iou', label: '위치' }, { code: 'no_kf', label: '없음' }];
const gs = L.missedGroups(objs, reasons, () => true);
check('원인별 묶음 (reasons 순서)', gs.map(g => [g.code, g.n]), [['merged', 1], ['low_iou', 1], ['no_kf', 2]]);
check('묶음마다 물체 번호', gs[2].objs, [0, 1]);
check('검출된 물체는 목록에 없다', gs.some(g => g.objs.includes(3)), false);
const gs2 = L.missedGroups(objs, reasons, o => o.kind !== 'human');
check('표시 필터를 반영', gs2.map(g => g.code), ['merged', 'no_kf']);

const kf = { gt: { tp: 2, split: 1, low_iou: 0, merged: 0, missed: 3 }, s3: { tp: 1, merged: 2, not_present: 5 } };
check('필터 없음', L.kfPass(kf, { st2: '', st3: '' }), true);
check('2D 과다분할 있는 keyframe', L.kfPass(kf, { st2: 'split', st3: '' }), true);
check('2D 과소분할 없는 keyframe', L.kfPass(kf, { st2: 'merged', st3: '' }), false);
check('3D 상태로도 거른다', L.kfPass(kf, { st2: '', st3: 'merged' }), true);
check('2D·3D 를 함께 만족해야', L.kfPass(kf, { st2: 'split', st3: 'low_iou' }), false);

// ---------------- m4 맵 빗금 (색 말고 무늬로도 구분)
check('채점 제외 아닌 판정은 꽉 채운다', ['detected', 'missed', 'merged'].every(st =>
  [[0, 0], [1, 0], [2, 3], [7, 5]].every(([x, y]) => L.hatch(st, x, y) === true)), true);
check('채점 제외는 2칸 칠하고 2칸 비운다 (45° 빗금)',
      [0, 1, 2, 3, 4, 5].map(x => L.hatch('ineligible', x, 0)), [true, true, false, false, true, true]);
check('대각선으로 이어진다 (x+y 기준)', L.hatch('ineligible', 2, 2), true);
check('음수 칸에서도 같은 주기 (4칸마다 반복)',
      [[-4, -4], [-1, -1], [-3, 0]].every(([x, y]) => L.hatch('ineligible', x, y) === L.hatch('ineligible', x + 8, y + 8)), true);
let on = 0, tot = 0;
for (let x = -20; x < 20; x++) for (let y = -20; y < 20; y++) { tot++; if (L.hatch('ineligible', x, y)) on++; }
check('칠하는 칸이 절반', on * 2 === tot, true);

// ---------------- 2차 리뷰: 그림 배율 · keyframe 고르기 · summary 경로
check('남는 자리에 맞춰 키운다 (640×480 → 1100×620)', Math.round(L.imgScale(1100, 620, 640, 480, 1) * 100) / 100, 1.29);
check('2배 이상이면 정수 배율 (가장자리 선명)', L.imgScale(1600, 1100, 640, 480, 1), 2);
check('확대 배율을 곱한다', L.imgScale(1600, 1100, 640, 480, 1.5), 3);
check('자리가 없으면 1', L.imgScale(0, 0, 640, 480, 1), 1);
check('최대 8배', L.imgScale(9000, 9000, 640, 480, 4), 8);

const obj = { rc: 'split', best_kf: 300, kfj: [{ kf: 236, st: 'ignored_match' }, { kf: 238, st: 'split' }, { kf: 239, st: 'missed' }] };
check('그 물체가 판정된 keyframe 으로 (대표 원인 먼저)', L.pickKf(obj, 0), 238);
check('이미 관련 있는 keyframe 이면 그대로', L.pickKf(obj, 239), 239);
check('판정 행이 없으면 가장 크게 보인 keyframe', L.pickKf({ best_kf: 7, kfj: [] }, 0), 7);
check('둘 다 없으면 지금 keyframe', L.pickKf({ best_kf: -1, kfj: [] }, 4), 4);

check('file:// 에서는 summary.json 을 부르지 않는다', L.canReadSummary('file:', '/x/viewer/uHumans2_a/index.html'), false);
check('뷰어 폴더만 서버로 열면 부르지 않는다 (404 방지)', L.canReadSummary('http:', '/uHumans2_office_s1_06h/index.html'), false);
check('상위 폴더를 서버로 열면 부른다', L.canReadSummary('http:', '/viewer/uHumans2_office_s1_06h/index.html'), true);
check('하위 경로에서도 규칙이 같다', L.canReadSummary('https:', '/runs/frontend_eval/viewer/uHumans2_a/index.html'), true);

// ---------------- m1 오래됨
const built = { steps: { score3d: 'a', seg2d: 'b' }, generated: '2026-09-18 03:00', generated_epoch: 100 };
check('같으면 최신', L.staleState(built, { steps: { score3d: 'a', seg2d: 'b' } }).state, 'fresh');
const st = L.staleState(built, { steps: { score3d: 'z', seg2d: 'b' } });
check('다르면 오래됨', st.state, 'stale');
check('어느 단계인지 말한다', st.steps, ['score3d']);
check('비교 대상이 없으면 알 수 없음', L.staleState(built, null).state, 'unknown');

console.log(fails ? `\n실패 ${fails}개` : '\n전부 통과');
process.exit(fails ? 1 : 0);
'''


def main():
    node = shutil.which('node') or shutil.which('nodejs')
    if not node:
        print('  SKIP node 가 없어 JS 논리 테스트를 건너뜁니다')
        return 0
    # 템플릿 안 <script> 구문 검사 — 따옴표 하나가 깨지면 화면 전체가 빈다 (node --check)
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for name in ('viewer_template.html', 'viewer_index_template.html'):
            src = (HERE / name).read_text()
            for i, b in enumerate(re.findall(r'<script(?![^>]*src=)[^>]*>(.*?)</script>', src, re.S)):
                f = Path(d) / f'{name}.{i}.js'
                f.write_text(b)
                r = subprocess.run([node, '--check', str(f)], capture_output=True, text=True)
                ok = r.returncode == 0
                bad += 0 if ok else 1
                print(f'  {"OK " if ok else "FAIL"} {name} script[{i}] 구문' + ('' if ok else ': ' + r.stderr.splitlines()[0][:160]))
    if bad:
        return 1
    html = (HERE / 'viewer_template.html').read_text()
    m = re.search(r'<script id="fblib">(.*?)</script>', html, re.S)
    if not m:
        print('  FAIL viewer_template.html 에 <script id="fblib"> 묶음이 없습니다')
        return 1
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        (t / 'fblib.js').write_text(m.group(1))
        (t / 'run.js').write_text(JS_TEST)
        r = subprocess.run([node, str(t / 'run.js')], cwd=str(t), capture_output=True, text=True)
        print(r.stdout.rstrip())
        if r.stderr.strip():
            print(r.stderr.rstrip()[-2000:])
        if r.returncode:
            print('  FAIL JS 논리 테스트 실패')
        return r.returncode


if __name__ == '__main__':
    sys.exit(main())
