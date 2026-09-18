# 감사 · 수정 이력 (frontend 평가)

이 벤치마크는 "표준 정의와 맞는가"를 독립 감사로 확인하고, 결함마다 테스트를 먼저 넣고 고치는 순서로 만들었습니다.
아래 근거 폴더는 워크스페이스 루트(`<WS>` = `$MERIDIAN_WS`) 기준이고, 폴더마다 실행 스크립트 · 출력 · `REPORT.md` 가 들어 있습니다.
현재 정의는 [METRICS.md](METRICS.md), 쓰는 법은 [../README.md](../README.md) 를 보세요.

## 한눈에

| 번호 | 날짜 | 무엇을 했나 | 근거 폴더 |
|---|---|---|---|
| 감사 A — 3D 검출 · 추적 | 26/09/17 | 표준 정의(TrackEval · MOT16 · Ristani · HOTA)와 대조. 결함 11건 — IDF1 · HOTA_α 계산, FP 규칙, 매칭 연속성, Frag, 부동소수 여유 | `runs/frontend_eval/_audit/A_3d_mot/` |
| 감사 B — 2D 분할 · 기하 | 26/09/17 | panopticapi · Tanks and Temples · KITTI · StarDist 와 대조. 결함 14건 — void 처리, 라벨 성분 채우기, `F@τ` 절차, 난이도 경계 | `_audit/B_2d_geometry/` |
| 감사 C — 파이프라인 · 캐시 · 뷰어 | 26/09/17 | 결함 18건(치명 3 · 중대 9 · 경미 6). "보인다"가 3D 난이도 · 2D 채점 대상 · 탐지 판정 창에서 서로 다르게 정의돼 있었고, 캐시가 코드·GT 변경을 놓쳤습니다 | `_audit/C_pipeline/` (`REPORT.md`) |
| 감사 D — 용어 | 26/09/17 | 같은 이름이 문서마다 다른 숫자였습니다 — "3D 재현율" 모집단, HOTA 를 단일 임계로 쓰면서 HOTA 라 부름, "놓침"이 5가지 뜻 | `_audit/D_terms/` (`REPORT.md`) |
| 수정 ① — 3D · 추적 | 26/09/17 | 감사 A 11건 + C3 을 표준대로 고침. `score_mot.py` IDF1 · HOTA_α · FP · 연속성 · Frag, 탐지 판정 창을 가시 프레임으로 | `_audit/fix_A_3d_mot/` |
| 수정 ② — 2D · 기하 | 26/09/17 | 감사 B 14건 + C10. GT 라벨 규칙(지지율 ≥ 0.5 · 5m 밖 픽셀 검증 · depth 0 은 void), θ 별 헝가리안, T&T `F@τ`, KITTI 등급 경계, 라벨 버전 기록 | `_audit/fix_B_2d/` |
| 수정 ③ — 파이프라인 · 캐시 · 뷰어 | 26/09/17 | 감사 C 15건. 캐시 키를 내용(sha1)으로, `eval.sh` 에 `-nt` 재실행과 정렬 게이트, 뷰어 색·이유를 한 규칙으로, 난이도 모집단을 채점 대상으로, 프레임 밀림 게이트(C17) | `_audit/fix_C_pipeline/` |
| 수정 ④ — 3D 매칭 | 26/09/18 | 3D 매칭을 keyframe 별 허용 거리 `IoU_τ ≥ 0.5` · 헝가리안으로 바꾸고 규칙 본문을 `match3d.py` 한 곳에 모음 | `_audit/fix_D_match3d/` |
| 수정 ⑤ — 연결 + E1~E4 | 26/09/18 | ③+④ 를 채점 · 뷰어 · 문서 · status.md 에 연결하고 E1~E4 를 고친 뒤 6개 시퀀스를 전부 다시 채점(27.6분) | `_audit/fix_E_integrate/` |
| 후속 F1~F3 | 26/09/18 | GPU 단계 허락 · Loc 하한 · status.md 형식 (아래 표) | `_audit/fix_E_integrate/followup/` |
| 사용성 리뷰 (ux_review) | 26/09/18 | 팀원 관점 독립 리뷰. 치명 4건(C1~C4) · 중대 10건(M1~M10) · 경미 7건(m1~m7). "지금 그대로 팀 저장소에 올리기에는 이르다" | `_audit/ux_review/` (`REPORT.md` · `logs/` · `shots/`) |
| 수정 ⑥ — CLI · 문서 · 용어 | 26/09/18 | 리뷰의 C1~C4 · M5~M10 · m5 · m6 (지금 작업): 명령·도움말·종료 코드, 이전 결과 보관과 `compare`, `terms.py` 한 곳, README·docs 재구성 | `_audit/fix_F_cli_docs/` |
| 수정 ⑦ — 뷰어 | 26/09/18 | 리뷰의 M1~M4 · m1~m4 · m7 (별도 작업): 맵 빈 화면, 클릭과 툴팁이 다른 물체를 고르는 문제, 미검출 목록·필터, 판정 근거 keyframe 으로 가기 | `_audit/fix_G_viewer/` |

## 수정 ⑤ 의 E1~E4

| 번호 | 무엇을 고쳤나 | 결과 |
|---|---|---|
| E1 덮음 규칙 | 과소분할 · 과다분할 판정의 "덮음"을 배타적(가장 가까운 GT 소유)으로. 20cm 허용을 그대로 쓰면 벽 예측 하나가 얇은 문틀 GT 들까지 덮은 것으로 셌습니다 | 과소분할 apartment 398 → 248건 |
| E2 중복 규칙 | TIDE Dupe 그대로 — 다른 예측에 매칭된 present GT 와 `IoU_τ ≥ 0.5` 일 때만 중복 | 중복 apartment 1335 → 333건 · office 6302 → 1049건 |
| E3 Loc 범위 | 3D 와 2D 가 같은 함수 `tide_rules.is_loc` 를 쓰고, 양쪽 경계를 포함 | 2D · 3D 판정이 같은 규칙 |
| E4 실제 데이터 경로 | 테스트가 읽는 실제 결과는 `FB_REAL_RUNS`, TrackEval 은 별도 경로로. `FB_RUNS` 를 샌드박스로 바꿔도 따라가지 않습니다 | 실제 데이터가 없으면 SKIP 으로 보고 |

E1~E3 은 **진단 수치만** 바꿉니다 — 같은 입력에서 매칭 · 트랙 재현율 · IDF1 · HOTA_α 는 그대로였습니다.

| 번호 | 무엇을 고쳤나 |
|---|---|
| F1 | frontend 를 다시 돌릴지 파일 시각이 아니라 출처 기록(`run_meta.json` `provenance`)으로 정합니다 — 코드 AST · frontend 소스 · 엔진 sha1 · 데이터셋 경로. GPU 단계는 `GPU=1` 이나 터미널 허락이 있어야 돕니다 (없으면 종료 코드 3) |
| F2 | Loc 하한을 우리가 정한 0.25 에서 TIDE 기본값 0.1 로 (tidecv 1.0.1 `quantify.py:428` 원문 확인, 사본 `_audit/fix_E_integrate/followup/refs/`). 원인 분류만 바뀌고 트랙 재현율 · PQ · IDF1 · HOTA_α 는 6개 시퀀스 모두 그대로 |
| F3 | status.md 를 승인받은 짧은 형식으로 되돌림 (둘째 줄 `<날짜> · <작성자>`, 글머리표 120자 이하) |

## 기존 결과 재채점

- **26/09/18 수정 ⑤**: 6개 시퀀스를 새 캐시 규칙으로 전부 다시 채점했습니다 — 정렬 검증 · 가시성 · 라벨 · 채점 전부 · 뷰어 · 판정 그림.
  frontend 는 다시 돌리지 않았고 `frontend_output.h5` 의 sha1 이 전후 같습니다. 이전 결과 사본은 `runs/frontend_eval/_audit/before_rescore/`
  (그 앞 단계 사본은 `_audit/before_fix/`) 에 있고, 걸린 시간은 6개 합계 27.6분입니다.
- **26/09/17 에 만든 6개 결과의 출처 기록**: 다시 돌리지 않고 근거를 확인해 채웠습니다 (`provenance.backfill`).
  `run_frontend.py` 의 AST 가 수정 전 사본과 같고, frontend 소스 24개 파일이 보관본과 바이트 단위로 같고, 엔진 sha1 · 데이터셋 경로가 기록과 같음을 확인했습니다
  (`_audit/fix_E_integrate/followup/f1_backfill.json`).
- **26/09/18 부터**는 채점 결과의 숫자가 바뀔 때마다 직전 결과가 `runs/frontend_eval/history/<시각>/` 에 자동으로 보관됩니다
  (같은 결과로 다시 채점하면 보관하지 않습니다)
  (`summary.json` · `summary.md` · `status.md` · `provenance.json`). `eval.sh history` · `eval.sh compare latest` 로 봅니다.

## 라벨 present 와 gt_vis 의 불일치

수정 ② 의 라벨 변경(5m 밖 픽셀 검증 · 지지율 0.5)으로 옛 gt_vis 대 2D 채점 대상 불일치가 1.7~1.8%p 줄었습니다
(감사 a04, `_audit/fix_C_pipeline/a04_sandbox_mismatch.json`). 남은 불일치는 0.7~1.6% 이고, 매 채점마다
`score.json` `keyframe_gt.presence_vs_vis` 에 기록됩니다 (표는 [METRICS.md](METRICS.md) "채점 대상 · 있음 세 정의").
정의를 하나로 합치는 것은 채점 로직 변경이라 하지 않았습니다.

## 엔진은 장비마다 다시 빌드한다 (이 프로젝트의 첫 실패)

받은 `.plan` (TensorRT 엔진)에는 **플랫폼 태그**가 박혀 있어 다른 장비에서 로드되지 않습니다
(x86 에서 `Platform specific tag mismatch`). 프로젝트 초기에 로봇(Jetson Orin · TensorRT 10.3)에서 받은 엔진을
이 PC(x86 · RTX 3060)에서 그대로 쓰려다 실패한 것이 첫 실패였습니다.

- 팀 저장소 릴리스 `frontend-engines-20260915` = **로봇용 Orin 엔진** (제목 "frontend engines (Orin, TensorRT 10.3)",
  릴리스 설명에도 "장비마다 다시 빌드할 것" 이라고 적혀 있습니다). 로봇에서는 그 설명대로
  `frontend/meridian_frontend/engines/` 에 둡니다. **x86 에서 내려받아 쓰면 안 됩니다.**
- 이 워크스페이스의 `models/frontend_rtx3060/{fastsam.plan, clip_image.plan}` 은 여기서 ONNX 로부터 다시 빌드한 것이고
  어디에도 올리지 않았습니다 (`build_engines.py` · `eval.sh engines`, 약 3.5분, FP16).
- 그래서 문서는 "엔진을 받는다" 가 아니라 "엔진을 빌드한다" 로 씁니다. 입력 ONNX 2개의 출처는
  [../README.md](../README.md) "준비 — TensorRT 엔진" 표에 있습니다.

## 그 밖의 기록

- 26/09/16 요청으로 segment 는 2D 이미지 비교, pointcloud 는 2D 맵 top-view 로 보게 되어 2D 분할 채점과 뷰어가 들어왔습니다.
- 26/09/17 결정 ④ 로 3D 매칭 기준이 "예측 복셀 중 τ 이내 GT 비율" 에서 허용 거리 `IoU_τ` 로 바뀌었습니다.
  예전 기준은 작은 순수 조각을 큰 조각보다 먼저 골랐습니다 (apartment kf29 / ep78: 192복셀 조각이 2403복셀 조각을 이김).
- `label window` (`[first − gap, last + gap]`) 는 26/09/17 부터 매칭에 쓰지 않습니다 (GT = keyframe 라벨 표면).
- 26/09/17 이전 결과의 옛 미검출 원인 값은 표에서 뺐습니다 — 옛 결과를 읽으면 `알 수 없는 이유 ‹값›` 으로 보이고,
  대응은 `score.json` `params.reason_mapping` 에 남아 있습니다.
