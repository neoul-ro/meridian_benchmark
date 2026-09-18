# 지표와 용어 정의 (frontend 평가)

`runs/frontend_eval/summary.md` · `status.md` · 뷰어에 나오는 모든 말의 정의입니다. 쓰는 법은 [../README.md](../README.md),
감사·수정 이력은 [HISTORY.md](HISTORY.md) 를 보세요. 경로는 워크스페이스 루트(`<WS>` = `$MERIDIAN_WS`) 기준입니다.

## 이 문서의 규칙

- **표시 이름 · 단위 · 소수 자리수 · 모집단은 `terms.py` 의 `METRICS` 표 한 곳에서 옵니다.** `summary.md` · `status.md` · 뷰어 · `compare` 가
  모두 이 표를 지나므로 같은 지표는 어디서나 같은 글자로 나옵니다 (종류마다 형식 하나 — 비율 1자리 % · 0~1 점수 2자리 · 거리 2자리 cm).
- 이름에 모집단을 넣습니다: `트랙 재현율 (Easy)` ≠ `트랙 재현율 (전체 채점 대상)`.
- **JSON·CSV 의 키와 값은 바꾸지 않았습니다.** 아래 "키 → 용어" 표로 대응시킵니다 (감사 D 결정).
- 한 낱말은 한 뜻으로만 씁니다.
  - **제외** = 채점 제외(ineligible), 즉 채점에 넣지 않는 GT 하나뿐입니다.
  - present 아닌 GT 에 매칭된 예측 검출은 **무시 GT 매칭(ignored match)** 입니다 (제외가 아니라 TP 도 FP 도 아닌 것).
  - 절반 넘게 void 에 떨어진 예측 마스크는 **void 겹침 무시** 입니다.
  - 평가 거리 범위 밖이라 뺀 점은 **범위 밖 제거** 입니다.
  - 과다분할(over-segmentation) · 과소분할(under-segmentation) 로만 부릅니다.
- **accuracy 는 기준이 둘이라 이름으로 구분합니다.**
  - `accuracy (GT 트랙 점군 기준)` — 예측 복셀에서 그 GT 트랙 점군 전체까지의 평균 거리 (`summary.md` 위치 장).
  - `accuracy (같은 keyframe GT 표면 기준)` — 예측 복셀에서 그 keyframe 에 실제로 보인 GT 표면까지의 평균 거리 (아래 "기하").

## 용어

| 쓰는 말 | 뜻 |
|---|---|
| GT 트랙(ground-truth track) | 한 GT 물체가 연속으로 보인 구간. GT h5 의 tracklet, 코드·CSV 의 "에피소드"(`ep_index`) |
| 예측 검출(predicted detection) | frontend 가 keyframe 하나에 publish(발행)한 관측 하나 (TrackletSet 의 원소) |
| 예측 ID(predicted ID) | frontend tracklet id |
| 가시(visible) | 640×480 crop · 평가 거리 범위 안에서 그 GT 트랙 픽셀이 1개 이상 |
| 평가 거리 범위(evaluation range) | 카메라에서 5m (GT 가 그 밖을 정의하지 않음) |
| 채점 대상(eligible) / 채점 제외(ineligible) | 채점에 넣는 것 / 넣지 않는 것. 3D 와 2D 의 기준은 다릅니다 (아래 "채점 대상 · 있음 세 정의") |
| void | 2D GT 라벨 0 (GT 트랙이 아닌 표면 · 검증 안 되는 픽셀) |
| GT pose(ground-truth pose) | 카메라 pose 로 GT 를 넣었습니다 (SLAM 오차 없음) |
| 트랙 재현율(track recall) | 채점 대상 GT 트랙 중, 있음(present)인 keyframe 에서 `IoU_τ ≥ 0.5` 로 매칭된 예측 검출이 1개 이상인 비율 |
| 미검출 트랙 | 위 조건을 못 채운 GT 트랙. 원인은 아래 "미검출 원인 값" |
| 허용 거리 IoU(tolerant IoU, IoU_τ) | `P·R / (P + R − P·R)`. 3D 매칭의 유사도 (아래 "3D 매칭 기준") |
| 있음(present) | (GT 트랙, keyframe) 쌍 중 그 keyframe 의 GT 라벨 픽셀(5m 안 · depth 유효)이 1600px 이상 |
| 무시 GT 매칭(ignored match) | present 아닌 GT 에 매칭된 예측 검출 — TP 도 FP 도 아님 (`detect_credit` = 0) |
| 검출 재현율(DetRe) | present 인 (GT 트랙, keyframe) 쌍 중 그 keyframe 예측 검출과 매칭된 비율. 정적 물체 값 = 추적표 CLEAR 의 TP/(TP+FN) |
| 중복 검출(duplicate) | 매칭 안 된 예측 검출 중, 같은 keyframe 에서 다른 예측 검출이 이미 매칭한 present GT 와 `IoU_τ ≥ 0.5` 인 것 (TIDE Dupe). 추적에서는 오검출(FP) |
| TIDE 오류(TIDE error) | 매칭 안 된 예측 검출마다 하나: 중복(Dupe) · Loc · 배경(Bkg) · 기타(Other) — TIDE `quantify.py:228-265` 순서 |
| P@20cm · R@20cm · accuracy | 거리 τ 는 cm 단위를 붙입니다. P = 예측 → GT 가 τ 안인 비율, R = GT → 예측이 τ 안인 비율, accuracy = 예측 → GT 거리 평균 |
| @IoU0.5 | IoU 임계는 IoU 를 붙입니다 |
| 과소분할 · 과다분할 · Loc(localization error) · 미검출(Miss) · 오검출(FP) | 2D · 3D GT 상태 `merged` · `split` · `low_iou` · `missed`, 예측 상태 `fp`. 미검출(Miss)은 FN 전체가 아닙니다 (FN = 넷의 합) |
| HOTA_α · DetA_α · AssA_α | α(유사도 임계) 한 점의 값. 논문의 최종 HOTA(α 0.05~0.95, 19점 평균)가 아닙니다 |
| MOTA | 놓친 것 · 잘못 낸 것 · ID 바뀜을 GT 검출 수로 나눠 1에서 뺀 값 (1이 만점, 음수도 나옵니다) |
| IDSW(ID switch) · Frag(fragmentation) · MT/PT/ML | MOT16 정의. MT/PT/ML = GT 트랙이 있는 keyframe 중 매칭된 비율 ≥ 80% / 20% 이상 80% 미만 / 20% 미만 |
| 이전 대응 유지(correspondence carry-over) | CLEAR MOT 매칭 1단계: 바로 이전 timestep 의 대응이 여전히 유효하면 유지 (MOT16 §4.1.1) |
| 데이터 연관(data association) | 시야를 벗어났다 돌아온 물체에 같은 ID 를 주는 일 (frontend 뒤 DA 모듈의 몫) |
| detect window · label window | 탐지 판정 창 / 복셀 라벨 창 — `score.json` `params.detect_window` · `params.label_window` |
| 가림 비율(occlusion ratio) · 최소 bbox 변 | 난이도 입력. 최소 bbox 변 = 보인 라벨 픽셀 bbox 의 min(w, h) |
| Easy / Moderate / Hard | KITTI 난이도 기준을 물체 일반에 맞춘 변형(adapted). 누적 (Moderate 는 Easy 포함) |
| GT 트랙당 예측 ID 수 | 한 번 이상 매칭된 GT 트랙마다 매칭된 서로 다른 예측 ID 개수의 평균 |
| timestep (MOT) | publish 된 keyframe 하나 |

### 미검출 원인 값 (`score_episodes.csv` `miss_reason`)

| 값 | 표시 이름 | 뜻 |
|---|---|---|
| `no_kf` | keyframe 없음 | GT 트랙이 있음(present)인 publish 된 keyframe 이 0개 |
| `merged` | 과소분할(under-segmentation) | 한 예측 검출이 이 GT 와 다른 GT 를 각각 절반 이상 배타적으로 덮음 |
| `split` | 과다분할(over-segmentation) | 복셀 절반 이상이 이 GT 소유인 예측 검출이 2개 이상, 그 조각들이 이 GT 를 절반 이상 배타적으로 덮음 |
| `low_iou` | Loc(localization error) | 매칭 안 됐고 최고 `IoU_τ` 가 0.1 이상 0.5 이하 (TIDE Loc 범위, 0.1 = TIDE 기본값) |
| `missed` | 기타 미검출(Miss) | present keyframe 이 있었지만 위 어디에도 해당 안 됨 |

트랙 원인 = present keyframe 마다의 상태 중 가장 많은 값, 동률이면 `merged` > `split` > `low_iou` > `missed`. 규칙 본문은 `match3d.py`.
표에 없는 값(26/09/17 이전 결과의 옛 값 포함)은 `summary.md` · 뷰어에 `알 수 없는 이유 ‹값›` 으로 그대로 보입니다 (조용히 빠지지 않습니다).
옛 값 대응은 `score.json` `params.reason_mapping` 에 남아 있습니다. 값 이름이 바뀌면 `terms.py` 의 `REASONS` 만 고칩니다.

### (keyframe, GT) 3D 상태 (`score_kf_gt.csv` `status`) · TIDE 오류 (`score_observations.csv` `tide_error`)

| 값 | 표시 이름 | 뜻 |
|---|---|---|
| `tp` | TP | present 이고 `IoU_τ ≥ 0.5` 로 매칭 |
| `merged` · `split` · `low_iou` · `missed` | 과소분할 · 과다분할 · Loc · 미검출(Miss) | present 인데 매칭 안 된 쌍 (위 표) |
| `ignored_match` | 무시 GT 매칭(ignored match) | present 아닌 GT 에 매칭 |
| `not_present` | present 아님 | 라벨 표면은 있지만 1600px 미만이고 매칭도 없음 — 채점하지 않음 |
| `tide_error` = `dupe` · `loc` · `bkg` · `other` | 중복(Dupe) · Loc · 배경(Bkg) · 기타(Other) | 매칭 안 된 예측 검출의 TIDE 오류 (FP / 무시 판정과는 따로). 매칭된 예측은 빈 값 |

### 우리가 정한 개념 (표준 대응어 없음)

| 개념 | 정의 |
|---|---|
| 트랙 재현율 | 위 표. 전체 · 가시 · 채점 대상 세 모집단으로 냅니다 |
| 채점 대상(3D) | 640×480 crop · 5m 안 가시 픽셀이 GT 트랙 동안 한 프레임이라도 1600px 이상. 1600px = frontend `sam.AREA_MIN` 256 proto 칸 × 2.5² |
| 있음(present) | (GT 트랙, keyframe) 쌍 중 그 keyframe 의 GT 라벨 픽셀(5m 안 · depth 유효) 1600px 이상. 3D 검출 · 추적(MOT) 공통 (`score_kf_gt.csv` `px_label_5m`) |
| 관측 매칭률 | 전체 예측 검출 중 GT 물체에 1:1 매칭된 비율. 분모에 GT 가 정의하지 않은 표면(벽·바닥)과 중복이 들어가서 precision 이 아닙니다 |
| 다물체 섞인 예측 검출 비율 | 매칭된 예측 검출 중, 가장 가까운 GT 표면이 2순위 GT 물체인 복셀이 `absorb_frac`(20%) 이상인 비율 |
| 2cm 복셀 정확일치율 | 매칭된 예측 검출의 복셀 중 활성 GT 트랙 복셀과 같은 2cm 칸에 있는 비율의 평균 |
| detect window | GT 트랙이 있음(present)인 keyframe — 이 안의 매칭만 탐지로 인정 (`detect_credit` = 1) |
| label window | `[first − gap, last + gap]`, gap = GT h5 `gap_frames`(5) = `params.window_margin_frames`. 26/09/17 부터 매칭에 쓰지 않습니다 (GT = keyframe 라벨 표면) |
| 2D 과소분할 (`merged`) | 매칭 안 된 GT 중, GT 2개 이상을 각각 50% 이상 덮는 예측 마스크에 덮인 것 (Hoover 과소분할의 50% 임계 변형) |
| 2D 과다분할 (`split`) | 매칭 안 된 GT 중, 50% 이상이 이 GT 안에 들어온 예측이 2개 이상이고 그 합집합이 GT 의 50% 이상을 덮는 것 |
| 2D · 3D Loc (`low_iou`) | 매칭 안 됐고 최고 IoU(3D 는 `IoU_τ`)가 `tide_rules.LOC_MIN_IOU`(0.1, TIDE 기본 background_threshold) 이상 매칭 임계(0.5) 이하 — 두 채점기가 같은 함수 `tide_rules.is_loc` |
| 가림 비율 | 화면 안에 투영된 GT 점 중 다른 표면이 `max(5cm, 3%·z)` 이상 앞에 있는 점 ÷ (보이는 점 + 가려진 점). KITTI 의 가림(0~3 범주)을 20% · 50% 경계로 환산한 것은 우리 선택 |
| 최소 bbox 변 | 위 표. KITTI 의 Min. bounding box height 를 min(폭, 높이) 로 바꿨습니다 |

## 채점 정의 (`score_frontend.py`)

GT = `src/meridian/meridian_benchmark/tracklets/<seq>.h5` (= `gt_graphs/*_hier.sparkdsg` 물체의 원본).
GT 트랙 = 한 물체가 연속으로 보인 구간, 점 = 그 구간에 본 표면 (2cm 복셀, 평가 거리 범위 5m 이내).

- **평가 거리 범위**: 예측 점 중 카메라 5m 초과는 버립니다 (GT 가 정의 안 함) → GT 와 같은 2cm 격자로 복셀화.
- **GT 표면**: keyframe 마다 그 keyframe 의 2D GT 라벨(`gt_labels_2d.py`, 값 = ep+1) 픽셀을 depth · K · GT pose 로 역투영, 5m 이내, 2cm 복셀
  (`gt_surface.py` — 3D 매칭과 기하 채점이 같은 코드). 라벨 폴더의 `meta.json` version · sha1 을 `score.json` params 에 남깁니다.
- **매칭**: 아래 "3D 매칭 기준". `summary.md` 머리말은 매 실행 `score.json` params 에서 그대로 옮깁니다 (하드코딩 없음).
- **미검출 트랙**: 있음(present)인 keyframe 에서 매칭된 예측 검출이 하나도 없음. 원인은 위 "미검출 원인 값".
- **채점 대상**: 640×480 crop 에서 1600px 이상 보인 적이 있는 GT 트랙 — frontend `sam.AREA_MIN`(256 proto 칸) 과 같은 크기.
  가시성은 `gt_visibility.py`: seg 색은 prefab 이라 "같은 색 ∩ GT 트랙 점군 6cm 이내 ∩ 5m" 픽셀만 셉니다
  (검증: 5m 제한을 풀면 GT 트랙별 보인 프레임 수 / GT 관측 프레임 수 = 중앙값 1.00, p5 0.95).
- **검출 재현율(DetRe)**: present 인 (GT 트랙, keyframe) 쌍을 그 keyframe 예측 검출이 잡은 비율 (publish 타이밍과 분리해서 본 검출 성능).
- **위치** (매칭된 예측 검출): `accuracy (GT 트랙 점군 기준)` · P@20cm · 2cm 복셀 정확일치율 · R@20cm(GT 트랙 표면 커버).
- **사람**: seg 색 `23d5ea` (사람 0명 시퀀스엔 없고 사람 있는 시퀀스에만 있는 유일한 색, 높이 1.5~1.6m).
  움직이는 사람은 GT 에서 여러 물체·GT 트랙으로 쪼개져 있어 따로 보고합니다.

### frontend 를 어떻게 넣었나 (`run_frontend.py`)

| 항목 | 값 | 이유 |
|---|---|---|
| 주입 방식 | `assemble(cfg, da_sink)` 로 조립 후 `on_info/on_pose/on_pair` 를 프레임 순서대로 직접 호출. 프레임마다 publish 완료 대기 | 입력 구독이 BEST_EFFORT depth=1 이라 ROS 재생은 처리 속도에 따라 프레임을 버립니다. 드랍 0 · 결정적. frontend 코드 무수정 |
| 해상도 | 720×480 → 가운데 640×480 crop (좌우 40px) | frontend 는 640×480 만 받습니다. 스케일 변화 없이 cx 360→320. frontend 기본 intrinsics 가 정확히 `[415.692, 415.692, 320, 240]` |
| rgb / depth | rgb8 / 16UC1 mm (언팩본 PNG 그대로) | preprocessing 계약 |
| pose | `world_T_left_cam` (optical), odom 을 이미지 stamp 로 보간, 이미지와 같은 stamp 로 주입 | `pose.py` = "stamp 이하 최신 1개, 보간 없음" |
| 엔진 | `FB_MODELS/frontend_rtx3060/` FP16, TensorRT 10.13 | fastsam = `FastSAM-s-1024.onnx` · clip = `clip_vit_b32_visual_pooled.onnx` (받은 엔진과 같은 patch-pool 구조, I/O 이름만 다름) |
| 저장 | `frontend_output.h5` — keyframe 마다 예측 검출(예측 ID, world 점, conf, embedding) | publish 되는 TrackletSet 전부 |

정렬 검증 (`check_alignment.py` + `eval.sh` 정렬 게이트 — 둘 다 통과해야 채점):

- 입력 정렬: frontend 가 publish 한 점 ↔ 같은 depth 를 독립 역투영한 점, p99 ≤ 3cm (실측 0.8cm).
- 프레임 번호 밀림 (감사 C17): 주장한 프레임 f 와 f±1 의 depth+pose 조밀 역투영에 frontend 점이 1mm 안에 겹치는 비율을 비교합니다.
  실측(keyframe 30개): 맞는 프레임 0.25 · 한 칸 밀린 프레임 0.02~0.06. 통과 조건 = f 에서 0.10 이상이고 f±1 의 2배 이상.
  정적 GT 점은 프레임과 무관한 world 좌표라 "GT 표면과 대조" 로는 통째 밀림을 볼 수 없어 depth 표면으로 대조합니다.
- 게이트는 `alignment_check.json` 에 기록된 `frontend_output.h5`(크기 · mtime, 다르면 sha1)가 지금 파일과 같을 때만 그 결과를 믿습니다.

## 3D 매칭 기준 (26/09/17 결정 ④ · 수정 ④ 반영 · 수정 ⑤ E1~E3)

규칙 본문은 `match3d.py` · `tide_rules.py` 한 곳이고, `score_frontend.py`(검출 · 원인)와 `score_mot.py`(추적 전처리 · 유사도)가 같이 씁니다.
원문 사본: `runs/frontend_eval/_audit/fix_D_match3d/refs/` (TIDE `tide_quantify.py` · `tide_main_errors.py`, mots_tools) ·
`runs/frontend_eval/_audit/B_2d_geometry/refs/` (T&T `tnt_evaluation.py`, panopticapi `pq_compute.py`) ·
`runs/frontend_eval/_audit/A_3d_mot/std/` (TrackEval · ScanNet).

| 항목 | 정의 | 출처 |
|---|---|---|
| 표본 | keyframe 하나에서 예측 i 의 2cm 복셀 `Pv_i`, GT j 의 표면 복셀 `Gv_j` (위 "GT 표면") | — |
| 허용 거리 P · R | `P_τ(i, j)` = `Pv_i` 중 `Gv_j` 까지 거리 < τ 인 비율 · `R_τ(i, j)` = `Gv_j` 중 `Pv_i` 까지 거리 < τ 인 비율. τ = 20cm (10 · 50cm 는 민감도), strict `<` (여유 1e-5m) | Tanks and Temples `evaluation.py:173-176` |
| IoU_τ | `P·R / (P + R − P·R)` = `F / (2 − F)`. τ → 0 이면 exact IoU \|∩\| / \|∪\| | ScanNet `evaluate_semantic_instance.py:119` · MOTS 마스크 IoU (TrackEval `kitti_mots.py:390`) |
| 매칭 | keyframe 마다 `IoU_τ ≥ 0.5` 인 (GT, 예측) 쌍만 후보, 헝가리안으로 IoU 합이 가장 큰 1:1. 시간 정보 없음 (시간 연관은 추적 채점) | TrackEval `clear.py:82-86` |
| 있음(present) | 그 keyframe 라벨 픽셀(5m 안 · depth 유효) ≥ 1600px (= frontend `sam.AREA_MIN` 256 proto 칸 × 2.5²). present 아닌 GT 도 매칭 후보이고, 거기 매칭된 예측은 무시 GT 매칭(ignored match) | TrackEval `mot_challenge_2d_box.py:374-384` (distractor 매칭 제거) |
| FP / 무시 | 매칭 안 된 예측의 (void 복셀 + 무시 GT 소유 복셀) / \|P\| > 0.5 → 채점에서 뺌, 아니면 FP (정확히 0.5 는 FP). 복셀의 주인 = 가장 가까운 GT 하나(동률 ±1e-9m 는 번호 작은 GT) | panopticapi `evaluation.py:155-163` · KITTI MOTS `kitti_mots.py:336-344` |
| 미검출 원인 | present 인데 매칭 안 된 (keyframe, GT): `merged` > `split` > `low_iou` > `missed` (이 순서로 처음 맞는 것). 트랙 = 최빈값 · present keyframe 없으면 `no_kf` | `score_2d` 규칙 · Hoover 1996 · TIDE `main_errors.py:24-27` · `78-81` |
| 덮음 (`merged` · `split`) [E1] | 배타적 — 가장 가까운 GT 소유(nearest-GT ownership). GT 복셀 y 는 예측 i 가 덮음 ⇔ y 의 최근접 예측 복셀(거리 < τ, 동률은 모두) 중 i 의 것이 있고 그 복셀의 주인이 이 GT. `split` 조각 = 복셀 절반 이상이 이 GT 소유인 예측. 매칭 `IoU_τ` 는 그대로 비배타 | 2D 픽셀 라벨의 배타성을 3D 로 옮김 (panopticapi 와 같은 소유 규칙) |
| Loc 범위 [E3 · F2] | `0.1 ≤ 최고 IoU ≤ 매칭 임계` (양쪽 포함). 0.1 = TIDE 기본 background_threshold. 2D `score_2d` 와 같은 상수·함수 `tide_rules.is_loc`. 최고 IoU 가 임계를 넘는데 매칭 안 된 GT(그 예측이 이웃 GT 에 쓰임)는 `missed` | TIDE `quantify.py:237` `bg_thresh <= iou <= pos_thresh` · `:428` `background_threshold:float=0.1` (tidecv 1.0.1) |
| 중복 [E2] | TIDE Dupe: 매칭 안 된 예측이 이 keyframe 에서 다른 예측에 매칭된 present GT 와 `IoU_τ ≥ 0.5`. Loc 범위 `[0.1, 0.5]` 가 먼저. 매칭 안 된 예측마다 `tide_error` = `dupe` · `loc` · `bkg` · `other` | TIDE `quantify.py:18` · `228-265` |

왜 이렇게 정했나 (자세한 경과는 [HISTORY.md](HISTORY.md)):

- **수정 ④**: 예전 매칭은 예측 복셀 중 τ 이내 GT 비율이 가장 큰 쪽을 골라서, 작은 순수 조각이 큰 조각을 이겼습니다
  (apartment kf29 / ep78: 192복셀 조각이 2403복셀 조각을 이김, completeness 1.9 → 71cm).
- **E1**: 20cm 허용을 덮음에 그대로 쓰면 벽 예측 하나가 20cm 안의 얇은 문틀 GT 들까지 "덮어" 과소분할이 부풀었습니다
  (`runs/frontend_eval/_audit/fix_D_match3d/fig/apartment_s1_00h_kf58_ep146_merged.png`, 전후 그림 `_audit/fix_E_integrate/fig/`).
- **E2**: 예전 중복 판정은 `P_τ` 가 가장 큰 GT 가 매칭돼 있는지만 보고 IoU 를 보지 않았습니다 (TIDE 와 다름).
- **E3 · F2**: Loc 하한이 3D 는 0.25 포함, 2D 는 0.25 미포함이었습니다. 지금은 둘 다 TIDE 의 `[bg, pos]` 양쪽 포함이고 하한은 TIDE 기본값 0.1 입니다.

## 2D 분할 채점 (`gt_labels_2d.py` · `score_2d.py`)

- **마스크**: `run_frontend.py` 가 keyframe 마다 SAM 마스크(192×256 격자, 칸 = 원본 2.5px)를 `obs/mask_bits` 로 저장합니다. TrackletSet 에는 없는 값입니다.
  publish 된 예측 검출과 같은 순서입니다 (검증, 감사 a13: 점의 96.4~97.0% 가 자기 마스크 안, 다음 예측 검출의 마스크엔 1.6~3.3%).
  "마스크 저장 전후 3D 출력이 비트 단위로 같다"는 옛 출력이 남아 있지 않아 지금은 다시 확인할 수 없습니다.
- **GT 라벨**: keyframe 프레임만 만듭니다. seg 색 영역을 연결 성분으로 나누고, 성분 픽셀의 3D 점이 어느 GT 트랙 점군(6cm)에 붙는지 투표합니다.
  지지율(hit/표본) ≥ 0.5 인 성분만 채우고, 5m 밖 픽셀은 같은 물체의 모든 GT 트랙 점군 6cm 이내일 때만 남기고, depth 0 픽셀은 void 입니다.
  나머지(벽·바닥 등 GT 에 없는 표면)는 void. 라벨 알고리즘 버전은 `meta.json` `version`.
- **채점**: θ 마다 IoU > θ 인 쌍만 남긴 행렬에서 헝가리안 1:1 (TP 수 최대 → IoU 합), θ = 0.25 / 0.5 / 0.75.
  IoU 는 void 칸을 예측에서 뺍니다 (panopticapi). Loc = 매칭 안 됐고 `0.1 ≤ 최고 IoU ≤ θ` (3D 와 같은 `tide_rules.is_loc`).
  채점 대상 = 라벨 면적 256칸(1600px) 이상. PQ = SQ × RQ (Kirillov et al. CVPR 2019).
  GT 상태 TP · 과소분할 · 과다분할 · Loc · 미검출(Miss), 예측 상태 TP · 무시 GT 매칭 · void 겹침 무시(void 비율 > 0.5) · 오검출(FP).

### 채점 대상 · 있음 세 정의 (3D gt_vis · 3D 라벨 · 2D 라벨)

| | 3D 채점 대상(eligible) — GT 트랙 | 3D 있음(present) — (GT 트랙, keyframe) | 2D 채점 대상 — (GT 트랙, keyframe) |
|---|---|---|---|
| 만드는 곳 | `gt_visibility.py` → `score_frontend` (`max_px_crop`) | `gt_labels_2d.py` 라벨 → `gt_surface` → `score_kf_gt.csv` `px_label_5m` | `gt_labels_2d.py` 라벨 → `score_2d` |
| 픽셀 | seg 색 ∩ 그 GT 트랙 점군 6cm 이내 ∩ 카메라 5m, stride 2 표본 × 4 | 성분 투표로 칠한 라벨 픽셀 중 depth 유효 · 광선 길이 5m 이내 | 라벨을 192×256 격자로 뽑은 칸 |
| 기준 | crop 가시 px ≥ 1600 인 프레임이 GT 트랙 동안 하나라도 | 그 keyframe 에서 ≥ 1600px | 그 keyframe 에서 ≥ 256칸 |

3D 에서 present 는 keyframe 라벨로, 채점 대상(트랙 재현율의 모집단)은 gt_vis 로 정합니다. 둘의 keyframe 단위 불일치는
`score.json` `keyframe_gt.presence_vs_vis` (라벨 present 와 gt_vis ≥ 1600 을 같은 keyframe 에서 비교)에 매번 기록됩니다:

| 시퀀스 | 라벨 present | gt_vis ≥ 1600 | 둘 다 | 라벨에만 | gt_vis 에만 | 불일치 비율 |
|---|---:|---:|---:|---:|---:|---:|
| apartment_s1_00h | 2246 | 2247 | 2231 | 15 | 16 | 1.4% |
| apartment_s1_01h | 2364 | 2365 | 2350 | 14 | 15 | 1.2% |
| apartment_s1_02h | 2358 | 2365 | 2343 | 15 | 22 | 1.6% |
| office_s1_00h | 3832 | 3847 | 3826 | 6 | 21 | 0.7% |
| office_s1_06h | 3942 | 3963 | 3938 | 4 | 25 | 0.7% |
| office_s1_12h | 4470 | 4488 | 4457 | 13 | 31 | 1.0% |

정의를 하나로 합치는 것은 채점 로직 변경이라 하지 않았습니다 (경과는 [HISTORY.md](HISTORY.md)).

## 난이도 (`gt_difficulty_2d.py`) — Easy / Moderate / Hard (KITTI 기준 변형)

- 잘림: GT 물체(같은 `gt_object_id` 의 모든 GT 트랙 점군 합집합)를 그 keyframe 카메라로 투영한 실루엣 중 640×480 밖 비율.
  카메라 뒤·3배 캔버스 밖 점도 화면 밖 면적으로 넣습니다.
- 가림 비율: 점 단위 depth 검사 — 같은 인스턴스 픽셀이면 보임, 다른 표면이 `max(5cm, 3%·z)` 이상 앞이면 가려짐. 사람은 점군이 이동 경로에 퍼져 있어
  계산하지 않고, 라벨이 화면 테두리에 닿으면 Easy 에서 뺍니다.
- 등급(누적, KITTI `evaluate_object.cpp:413` 경계와 같게): Easy = 최소 bbox 변 **> 40px** · 잘림 ≤ 15% · 가림 비율 ≤ 20% /
  Moderate = > 25px · ≤ 30% · ≤ 50% / Hard = > 25px · 잘림 ≤ 50% / 그 밖 = 등급 없음.
  `summary.md` 의 기준 문구는 `level_of` 를 경계값으로 찔러 읽은 값입니다.
- **난이도별 트랙 재현율(3D)** (감사 C1): 모집단 = 3D 채점 대상 GT 트랙(가시 1600px 이상)만. 등급 = 그 GT 트랙의 keyframe 인스턴스 중
  **2D 채점 대상(라벨 면적 ≥ 256칸)** 인 것의 가장 쉬운 등급.

## 기하 (`score_geometry.py`)

매칭된 예측 검출을 **같은 keyframe 에 실제로 보인 GT 표면**(GT 라벨 픽셀 역투영, stride 1 — `gt_surface.py`, 3D 매칭과 같은 코드)과 비교합니다.
매칭 열은 `score.json` `params.tau_m` 의 `match@<τcm>` 을 읽습니다. 무시 GT 매칭(`detect_credit` = 0)은 채점하지 않고 `skipped_obs.ignored_match` 로 셉니다.

- `accuracy (같은 keyframe GT 표면 기준)` · `completeness` (최근접 거리 평균) · `Chamfer-L1` = (accuracy + completeness)/2 (비제곱)
  — 이름과 정의 모두 Occupancy Networks(Mescheder et al. 2019) `eval.py`.
- `F@τ` = `2·P@τ·R@τ/(P@τ+R@τ)`, τ = 5 · 10 · 20cm, **d < τ** — Tanks and Temples 절차 (두 점군 τ/2 복셀 다운샘플).
- 완벽 예측 상한(GT 라벨 픽셀을 frontend 격자 밀도로 넣은 값)을 같이 기록합니다. 건너뛴 매칭 예측 검출은 이유별로 `skipped_obs` 에 셉니다.
- GT 트랙 전체 점군과 비교하면 한쪽 면만 본 예측 검출의 completeness 가 부당하게 나빠서 따로 뒀습니다.

## 추적 (`score_mot.py`) — keyframe 단위 MOT, 정적 물체

- timestep = publish 된 keyframe. GT 트랙 = 정적 물체의 GT 트랙. 위키 Evaluation "보낸 거 또 보냄 → DA 가 해결" 에 맞춰
  시야를 벗어났다 돌아와 새 예측 ID 를 받는 건 세지 않습니다 (데이터 연관의 몫). 시퀀스 전체 물체 기준은 참고용으로 같이 냅니다.
- 있음(present) = 그 keyframe 라벨 5m 안 1600px 이상 (`score_kf_gt.csv`, 3D 검출과 같은 정의). 유사도 = `IoU_τ` (`gt_iou@20` 열).
- 예측 검출 = keyframe 마다 3D 와 같은 매칭(`match3d.match`)으로 전처리한 뒤, 무시 GT(present 아님 · 사람)에 매칭된 관측과
  (void + 무시 GT 소유 복셀) / |P| > 0.5 인 관측을 뺀 나머지. **매칭 안 된 예측 검출은 전부 오검출(FP)** (중복 포함, TrackEval `clear.py`, MOT16 §4.1.3).
  뺀 관측은 `n_obs_excluded` (TrackEval `mot_challenge_2d_box.py:370-387` distractor 제거 · `kitti_mots.py:336-344` 무시 영역).
- 매칭 = CLEAR MOT: 바로 이전 timestep 의 대응이 유효하면 유지(이전 대응 유지) + `IoU_τ`, TrackEval 과 같은 절차
  (`test_trackeval_parity.py` 가 1e-6 이내로 대조합니다).
- IDSW · Frag · MT/PT/ML = MOT16 (MT 경계 ≥ 80%, TrackEval 은 > 0.8 → `MT_ratio_eq_80`). IDF1 = Ristani 2016. HOTA_α = α 한 점 (TrackEval `hota.py` 매칭).
- 사람 제외: 움직이는 사람이 GT 에서 여러 물체 번호로 쪼개져 있어 GT 정체성부터 믿을 수 없습니다.

## 키 → 용어

| 파일 · 키 | 표준 용어 · 정의 |
|---|---|
| `score_episodes.csv` `ep_index` · `gt_tracklet_id` · `gt_object_id` | GT 트랙 번호 · GT h5 tracklet id · GT 물체 번호 (같은 물체의 GT 트랙들이 공유) |
| `score_episodes.csv` `detected` · `detected@<τcm>` | 트랙 검출 여부 (τ 별) |
| `score_episodes.csv` `miss_reason` | 미검출 원인 (위 표) |
| `score_episodes.csv` `max_px_crop` · `vis_frames_crop` · `frames_px_ge_min` | 전 프레임 최대 crop 가시 px · 가시 프레임 수 · 1600px 이상 프레임 수 |
| `score_episodes.csv` `n_kf_in_window` · `n_kf_while_visible` | `[first, last]` 안 keyframe 수 · 가시인 동안 keyframe 수 |
| `score_episodes.csv` `n_matched_obs` · `n_matched_obs_no_credit` · `n_absorbed_obs` | 탐지 인정 매칭 예측 검출 수 · 무시 GT 매칭 수 · deprecated: `merged` 인 keyframe 수 |
| `score_episodes.csv` `n_kf_with_surface` · `n_kf_present` · `n_kf_<상태>` · `best_iou` | 라벨 표면 있는 keyframe 수 · present keyframe 수 · 상태별 keyframe 수 · present keyframe 에서의 최고 `IoU_τ` |
| `score_episodes.csv` `coverage_at_tau` | R@τ (GT 트랙 점군 기준) |
| `score_observations.csv` `obs` · `tracklet_id` | 예측 검출 번호 · 예측 ID |
| `score_observations.csv` `match@<τcm>` · `dup_of` · `dup_of_ep` · `detect_credit` | 매칭된 GT 물체 번호 (0 = 없음) · 중복 검출(TIDE Dupe)이면 그 물체 · 그 GT 트랙 · present GT 매칭이면 1 (무시 GT 매칭 0) |
| `score_observations.csv` `gt_counts@<τcm>` · `gt_vox@<τcm>` · `gt_iou@<τcm>` · `n_vox` | 가장 가까운 GT 표면이 그 물체인 복셀 수 · 같은 수의 GT 트랙별 · GT 트랙별 `IoU_τ` (17 유효숫자) · 5m 안 복셀 수 |
| `score_observations.csv` `match_iou@<τcm>` · `match_P@<τcm>` · `match_R@<τcm>` · `pred_status` · `excluded_reason` · `void_frac@<τcm>` · `ignore_frac@<τcm>` · `tide_error` | 매칭 쌍의 `IoU_τ` · `P_τ` · `R_τ` · `tp` / `ignored_match` / `fp` / `excluded` · 그 이유 · void 비율 · void + 무시 GT 비율 · TIDE 오류 |
| `score_kf_gt.csv` `px_label_5m` · `present` · `vis_px_crop` · `status` · `matched_obs` · `match_iou` · `best_iou` · `best_obs` · `best_P` · `best_R` · `n_frags` · `split_cover` · `merged_obs` · `matched@<τcm>` | 5m 안 라벨 px · present 여부 · gt_vis crop px (대조용) · 3D 상태 · 매칭 관측 · 그 `IoU_τ` · 최고 `IoU_τ` · 그 관측 · 그 `P_τ` · `R_τ` · 배타 조각 수 · 배타 합집합 덮음 · 과소분할 만든 관측 · τ 별 매칭 여부 |
| `score_observations.csv` `dist_med_cm` · `dist_mean_cm` · `dist_p90_cm` · `within_tau` · `voxel_exact` | 예측 → GT 트랙 점군 거리(accuracy 계열) · P@τ · 2cm 복셀 정확일치율 |
| `score.json` `missed.n_gt_episodes` · `missed.recall` | GT 트랙 수 · 트랙 재현율(전체) |
| `score.json` `visibility.n_in_view` · `recall_in_view` | 가시 GT 트랙 수 · 트랙 재현율(가시) |
| `score.json` `visibility.n_detectable` · `recall_detectable` · `miss_reasons_detectable` | 채점 대상 GT 트랙 수 · **트랙 재현율** · 미검출 원인 |
| `score.json` `visibility.keyframe_level.recall` | 검출 재현율(DetRe) — CLEAR TP/(TP+FN) |
| `score.json` `accuracy.obs_match_rate` · `duplicate_rate` · `multi_object_obs_rate` | 관측 매칭률 · 중복 검출 비율 · 다물체 섞인 비율 |
| `score.json` `accuracy.n_obs_tp` · `n_obs_ignored_match` · `n_obs_fp` · `n_obs_excluded` · `excluded_reasons` · `unmatched_tide_errors` · `unmatched_tide_errors_by_status` | 예측 검출 상태별 수 · 그 이유 · TIDE 오류 분포 (FP / 무시별) |
| `score.json` `keyframe_gt` `n_present` · `n_tp` · `recall` · `status` · `presence_vs_vis` | present 쌍 · TP · 검출 재현율 · (keyframe, GT) 상태 개수 · 라벨 present 와 gt_vis ≥ 1600 불일치 |
| `score.json` `accuracy.dist_error_cm.per_obs_mean` · `within_tau_fraction_mean` · `episode_coverage_at_tau` · `voxel_exact_2cm_mean` | `accuracy (GT 트랙 점군 기준)`(관측별 평균의 분위수) · P@τ · R@τ(GT 트랙별 분위수) · 2cm 복셀 정확일치율 |
| `score.json` `missed.object_recall` | 물체 재현율 (GT 물체 중 GT 트랙 하나라도 검출) |
| `score.json` `params.window_margin_frames` · `detect_window` · `label_window` | gap · detect window · label window |
| `score.json` `params.similarity` · `iou_threshold` · `min_frac` · `loc_min_iou` · `labels_dir` · `labels_version` · `labels_meta_sha1` · `presence` · `fp_rule` · `dup_rule` · `tide_error` · `merge_split_coverage` · `miss_reasons` | 3D 매칭 정의 (위 절) · `min_frac` = `iou_threshold` 옛 이름 · 쓴 라벨의 출처 |
| `score_mot.json` `metrics` · `metrics_object` | GT 트랙 단위 · 물체 단위 (timeline 의 `episode` · `object`) |
| `score_mot.json` `n_gt_tracks` · `n_pred_ids` · `gt_detections` · `pred_detections` | GT 트랙 수 · 예측 ID 수 · GT 검출 수(= TP + FN, TrackEval `GT_Dets`) · 예측 검출 수 |
| `score_mot.json` `TP` · `FN` · `FP` · `IDSW` · `Frag` · `MT` · `PT` · `ML` · `MT_ratio_eq_80` | CLEAR MOT (TrackEval `CLR_TP` …) · MT 경계 차이 수 |
| `score_mot.json` `IDF1` · `IDTP` · `IDFP` · `IDFN` · `IDR` · `IDP` | Identity (Ristani 2016) |
| `score_mot.json` `HOTA` · `DetA` · `AssA` | HOTA_α · DetA_α · AssA_α |
| `score_mot.json` `HOTA_TP` · `HOTA_FN` · `HOTA_FP` · `DetRe` · `DetPr` · `AssRe` · `AssPr` | HOTA α 매칭 기준 값 — 이 `DetRe` 는 위의 검출 재현율(CLEAR)과 다를 수 있습니다 |
| `score_mot.json` `mean_ids_per_track` · `n_obs_excluded` | GT 트랙당 예측 ID 수 · 예측 검출에서 뺀 관측 수 (`no_voxels_within_range` · `matched_non_present_gt` · `matched_human_gt` · `mostly_ignore_region`) |
| `score_2d.json` `all` · `static` · `human` · `n_gt_instances` · `recall["θ"]` · `precision` | 전체 · 정적 · 사람 · 2D 채점 대상 GT 인스턴스 수 · 재현율 @IoUθ · 정밀도 @IoU0.5 |
| `score_2d.json` `SQ` · `RQ` · `PQ` · `n_tp` · `n_fp` · `n_fn` · `mean_best_iou_fn` · `params.loc_min_iou` | Panoptic Quality 구성 · FN 의 최고 IoU 평균 · Loc 하한 |
| `score_2d.json` `gt_status` · `pred_status` · `by_level` · `recall_by_size` | 2D GT 상태 · 예측 상태 (위 용어) · Easy/Moderate/Hard · 면적 구간(칸 = 2.5×2.5px) |
| `score_2d_gt.csv` `eligible` · `area_cells` · `trunc` · `occ` · `min_dim` · `level` | 2D 채점 대상 · 라벨 면적(칸) · 잘림 · 가림 비율 · 최소 bbox 변 · 0 Easy 1 Moderate 2 Hard 3 등급 없음 |
| `score_2d_pred.csv` `status` = `tp` · `tp_small` · `ignore` · `fp` | TP · 무시 GT 매칭(ignored match) · void 겹침 무시 · 오검출(FP) |
| `score_geometry.json` `summary` `accuracy_cm` · `completeness_cm` · `chamfer_l1_cm` · `P@τ` · `R@τ` · `F@τ` | accuracy (같은 keyframe GT 표면 기준) · completeness · Chamfer-L1 (OccNet) · P@τcm · R@τcm · F@τcm (Tanks and Temples) |
| `score_geometry.json` `perfect_upper_bound` · `skipped_obs` · `params.match_column` | 완벽 예측 상한 · 건너뛴 매칭 예측 검출 (이유별: `ignored_match` 등, `outside_detect_window` = 같은 값의 deprecated 별칭) · 읽은 매칭 열 |
| gt_vis npz `px_crop` · `px_full` | GT 트랙 · 프레임별 crop(640) · 원본(720) 가시 px |
| `summary.json` (report 가 더함) `counts_3d` · `level_3d{n, n_detected, recall}` · `level_3d_criteria` · `report_steps` | 트랙 재현율 분자·분모 · 난이도별 트랙 재현율 · Easy/Moderate/Hard 기준(코드에서 읽음) · 단계별 캐시 키·완료 시각 |
| `tests_summary.json` `files.<파일>` · `totals` | status(pass/fail/skip) · n_ok · n_fail · n_skip · skips(건너뛴 이유) · 로그 경로 · 합계 |

## status.md 판정 기준 (`status_md.py`)

팀 공유용 요약은 짧게 씁니다: 제목 · `<날짜> · <작성자>`(`FB_AUTHOR`) · 소개 2줄 · 결과 표 · 글머리표 4개 · 뷰어 한 줄 · 한계 한 줄 · 끝 질문(`FB_CLOSING`).
글머리표는 **위치** · **분할** · **추적** 한 줄씩(숫자 + 판정 한 마디)과 용어 한 줄이고, 각 120자 이하입니다.
판정 문구는 `status_md.JUDGE` 에 적힌 기준으로 고릅니다 — 표준 등급이 아니라 읽기용으로 우리가 정한 기준입니다.
좋다고 말하려면 가장 나쁜 시퀀스도 기준을 넘어야 하고, 판정은 화면에 보이는 반올림 숫자로 합니다.

| 글머리표 | 보는 값 | 문구 |
|---|---|---|
| 위치 | F@20cm 최솟값 | ≥ 0.90 "표면이 GT 와 잘 겹칩니다" · ≥ 0.75 "대체로 겹칩니다" · 그 밖 "많이 어긋납니다" |
| 분할 | PQ 최솟값 | ≥ 0.70 "분할이 좋습니다" · ≥ 0.50 "보통입니다" · 그 밖 "아직 약합니다" |
| 추적 | GT 트랙당 예측 ID 수 최댓값 | ≤ 1.2 "거의 유지됩니다" · ≤ 1.5 "가끔 바뀝니다" · 그 밖 "자주 바뀝니다" |

뷰어의 물체 색과 이유도 한 규칙에서 나옵니다 — 채점 대상 GT 트랙들의 미검출 원인 중 가장 많은 값(`terms.pick_reason`)을 고르고,
그 값의 분류 색(`terms.COLORS`: 검출 · 미검출 · 과소분할 · 채점 제외)을 씁니다. 판정 그림(`render_examples.py`)도 같은 색·같은 이름을 씁니다.
0~1 점수(PQ · IDF1 · HOTA_α · F@20cm)는 뷰어·`summary.md`·`status.md`·`compare` 모두 **소수 2자리**입니다 (`terms.METRICS` 의 `score` 종류).

뷰어 필터는 **맞지 않는 물체를 지우지 않고 흐리게** 그리고 "맞는 물체 N / 전체 M" 개수를 함께 보입니다.
아예 지우려면 "맞지 않는 물체를 아예 숨기기" 체크 상자를 켭니다 (사용성 리뷰 M3). keyframe 탭에는 2D · 3D 상태 필터가 따로 있습니다.
색이 겹치는 자리는 선 모양(실선 · 점선)과 이름 · 개수로 구분합니다 — 회색 "채점 제외" 와 청록 "과소분할" 은
색약(deutan) 판정에서 ΔE 2.0 으로 가까워서, 색만으로 구분하지 않도록 범례에 모양과 개수를 같이 둡니다.

## 자체 검증 (`eval.sh test` → `run_tests.py`)

`frontend_benchmark/tests/test_*.py` 를 전부 파일마다 따로 돌리고, 원문 로그를 `<LOGS>/tests/` 에 남기고, 실패하면 traceback 끝부분을 보여 줍니다.
요약은 `<RUNS>/tests_summary.json` (파일별 pass/fail/skip · OK 수 · SKIP 줄과 이유). 실제 데이터가 필요한 테스트는
`paths.REAL_RUNS`(`FB_REAL_RUNS`)와 TrackEval 경로에서 읽습니다 — `FB_RUNS` 를 샌드박스로 바꿔도 따라가지 않습니다.
실제 데이터가 없으면 `SKIP <이유>` 를 찍고 종료 0 입니다.

| 파일 | 무엇 |
|---|---|
| `test_match3d.py` · `test_gt_surface.py` | 3D 매칭 규칙(`IoU_τ` · 배타 덮음 · TIDE 규칙 · 전체 거리 행렬 대조) · keyframe GT 표면 |
| `test_score.py` · `test_score_mot.py` · `test_trackeval_parity.py` | 3D 채점 · 추적 지표 · TrackEval 대조 |
| `test_score_2d.py` · `test_gt_labels_2d.py` · `test_gt_difficulty_2d.py` · `test_score_geometry.py` | 2D 분할 · GT 라벨 · 난이도 · 기하 |
| `test_report_cache.py` | `report.py` 캐시 시나리오 (GT 교체 · 다른 GT 폴더 · 모듈 변경 · 라벨 코드 변경 · params 덮임 · 변경 없음 · 누락 시퀀스) |
| `test_pipeline.py` · `test_terms_docs.py` | 단계 그래프 · 한 번만 반올림 · 머리말 params · 문서 문구 · 용어 한 곳(`terms.py`) · README 구성 |
| `test_cli.py` · `test_eval_sh.py` · `test_report_cli.py` | 명령·이름 해석·종료 코드 · `-nt` 재실행 · 정렬 게이트 · GPU 허락·계획 (가짜 frontend, GPU 안 씀) |
| `test_frontend_provenance.py` · `test_check_alignment.py` | frontend 출처 기록 · 프레임 번호 ±1 밀림을 게이트가 잡는가 (실제 h5, 약 20초) |
| `test_status_md.py` · `test_viewer.py` · `test_viewer_js.py` | status.md 의 모든 칸을 따로 계산해 대조 · 뷰어 자료와 화면 규칙 |

## 알려진 한계

- 엔진은 이 PC 에서 다시 빌드한 것입니다 (FP16). Orin 원본 엔진과 수치가 완전히 같다고 보장할 수 없습니다.
- 입력은 드랍 0 입니다 — 프레임을 버릴 수 있는 실시간 배포보다 좋은 조건입니다.
- uHumans2 카메라는 office 16.8Hz · apartment 12.9Hz 입니다. frontend 상수(`KF_MIN_GAP` 5프레임 등)는 30Hz 기준입니다.
- GT 는 두께 4cm 이하 표면·천장 띠를 제외했습니다 → office 벽·바닥·천장 예측 검출은 "매칭 없음"으로 나옵니다 (오검출이 아닙니다).
- office 에서 frontend 자체 경고 `overflow: n_conf=… n_cand=…` (n_conf ≥ K1 또는 n_cand ≥ LANES) 가 프레임의 1~2% 에서 납니다
  — `summary.md` 실행 정보의 overflow 경고 프레임을 보세요.
- 카메라 pose 는 GT pose 입니다. SLAM 오차가 빠진 frontend 단독 성능입니다.
- 3D 채점 대상은 면적(1600px) 기준이라 화면 끝 얇은 띠도 들어갑니다 → Easy/Moderate/Hard 로 따로 채점해 보완합니다.
- 데이터셋(`FB_DATA`)은 내용 해시를 하지 않고 경로만 캐시 키에 넣습니다 (이미지 수만 장). 데이터를 제자리에서 바꾸면 `--force`.
- 3D 매칭은 20cm 허용이라 얇은 GT 가 큰 평면 예측과 `IoU_τ ≥ 0.5` 로 매칭될 수 있습니다 (배타 덮음은 원인 진단에만 쓰고 매칭에는 쓰지 않습니다).
- 중복 · Loc · Bkg 는 TIDE 순서를 따르지만, TIDE 의 점수순 탐욕 매칭 대신 헝가리안 매칭 결과에 적용합니다 (Bkg 경계는 TIDE 기본 0.1).
- 2D GT 라벨은 keyframe 프레임만 만듭니다. keyframe 이 아닌 프레임의 성능은 이 벤치마크로 알 수 없습니다.
- 3D present 와 3D 채점 대상(gt_vis)의 정의가 달라 keyframe 단위로 0.7~1.6% 불일치가 남아 있습니다 (위 표).
