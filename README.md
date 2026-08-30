# meridian_benchmark

Meridian 파이프라인([neoul-ro/meridian](https://github.com/neoul-ro/meridian)) 벤치마크 하네스.
계획: [docs/BENCHMARK_PLAN.md](docs/BENCHMARK_PLAN.md)

데이터: uHumans2 (MIT SPARK Lab), `~/yun/meridian_ws/datasets/uHumans2/` 언팩본.
GT 출력: `~/yun/meridian_ws/datasets/gt/<sequence>/`.

## 모듈

단일 ament_python 패키지 `meridian_benchmark`:

| 모듈 | 역할 | ROS |
|---|---|---|
| `uhumans2.py` | uHumans2 언팩본 리더 (index, 이미지, odom 보간 → `world_T_camera`) | 무관 |
| `gt_build.py` / `gt_verify.py` | 오프라인 GT builder / 검증기 (CLI) | 무관 |
| `metrics.py` / `score.py` | 채점 프리미티브 / 채점기 CLI (`bench-score`) | 무관 |
| `runner.py` | N회 실행 + 중앙값 리포트 (`bench-run`) | launch 호출 |
| `player.py` | GT → 입력 토픽 실시간 발행 노드 (구독자 대기 후 시작) | 노드 |
| `recorder.py` | 입·출력 토픽 수신 시각 + payload 기록 노드 | 노드 |
| `launch/*.launch.py` | 모듈별 벤치 launch 8종 (player+모듈+recorder) | launch |

## GT 생성

```bash
cd src/meridian_benchmark

# 빌드 (시퀀스당 1회. 기본: stride 4, voxel 2cm, 승격 = 3프레임 이상에서 100px 이상)
python3 -m meridian_benchmark.gt_build \
  --dataset ~/yun/meridian_ws/datasets/uHumans2/apartment_scene/uHumans2_apartment_s1_00h \
  --out ~/yun/meridian_ws/datasets/gt

# 검증 (A 구조 / B 일관성 / C 자기 재투영 / D 관측↔객체 cloud 일관성(8 m 이내 점) / E 인스턴스 분리)
python3 -m meridian_benchmark.gt_verify \
  --dataset ~/yun/meridian_ws/datasets/uHumans2/apartment_scene/uHumans2_apartment_s1_00h \
  --gt ~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h

# DA용 GT tracklet (시퀀스당 1회; <gt>/gt_tracklets.h5 + gt_tracklets_index.csv)
#   tracklet = 객체의 연속 가시 구간 (관측 공백 ≤5프레임은 연결), 시야에서 사라지는 프레임에 발행,
#   점군 = 그 구간에 카메라가 본 표면만 2cm map-frame 복셀(셀 중심), semantics = upstream clip 노드
#   파이프라인(ViT-B/32 mask_weighted_value)을 관측마다 오프라인 실행해 평균. meridian_clip 모델 필요.
source install/setup.bash   # rclpy + meridian_clip
python3 -m meridian_benchmark.gt_tracklets \
  --gt ~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h \
  --dataset ~/yun/meridian_ws/datasets/uHumans2/apartment_scene/uHumans2_apartment_s1_00h
#   --gap 5 --voxel 0.02 --backend tensorrt|torch --no-embed
# 파일 계층은 meridian_msgs/Tracklet 필드 그대로 (tracklets/<id>/…), wire 외 정보는 _metadata/ (voxel별 카메라
# RGB는 _metadata/point_rgb), 파일 수준 index/ 는 columnar 조회용. 16px 미만 관측뿐인 tracklet은 semantics가
# 빈 배열 (n_obs_embedded=0). 관측 시점 카메라에서 5m 초과인 점은 버림 (--max-range 5; 개수는
# _metadata/n_points_far). 복셀 extent가 x/y/z 중 한 축이라도 4cm 이하(복셀 1~2층)인 tracklet은 제외
# (--min-extent 0.04; 시퀀스당 절반 이상). 모든 점이 천장(GT 지도 z-히스토그램 최상단 밀집 슬래브, 시퀀스 상수)
# 아래 30cm 안에 있는 tracklet(조명·스프링클러)도 제외 (--ceiling-margin 0.3; 단층 office 기준, apartment는 최상층
# 천장만 잡힘). 카메라 거리(프레임별 최소의 min, 프레임별 평균의 mean)는
# _metadata/dist_min_m, dist_mean_m 및 index/csv 컬럼. --recolor 는 기존 h5에 point_rgb만 다시 계산,
# --filter 는 기존 h5에 --min-extent/--ceiling-margin 만 적용(id 재부여; 거리 필터는 재빌드 필요).

# tracklet 시각화 (office 00h): 큰 top-down 이미지 — 지도 전체 회색(천장은 z 필터로 제거), 해당 tracklet은
# 카메라 RGB 색으로 칠하고 빨간 상자로 표시. --dataset 을 주면 지도가 없는 빈 구석에 대표 프레임(구간 내
# 최다 픽셀 관측)의 rgb 이미지와 class(seg_cam 색상) 이미지를 마스크 윤곽·상자와 함께 삽입.
# <gt>/viz/tracklet/tracklet_<id>_obj<gt_object_id>.png
python3 -m meridian_benchmark.gt_viz --gt ~/yun/meridian_ws/datasets/gt/uHumans2_office_s1_00h --view tracklet \
  --dataset ~/yun/meridian_ws/datasets/uHumans2/office_scene/uHumans2_office_s1_00h
#   --range A B (tracklet id), --zmax (기본: 천장 슬래브 30cm 아래), --px-cm (기본 1 또는 2)
# 발행 토픽/타입은 미정 — 현재는 h5 저장까지. VS Code에서는 H5Web 확장으로 열어봄.
```

알려진 identity 한계: 서로 **실제로 접촉하는** 동일 prefab 객체들은 하나의 GT 인스턴스로 병합된다
(예: office 회의실 의자 6개 = oid 하나). 붙어 있는 같은 색 표면 사이에는 경계 증거가 데이터에 없어
원리적으로 분리 불가 — GT identity는 "객체 군집" 단위로 일관되므로 채점은 성립한다. 화면에서만 겹쳐
보이고 3D에서는 떨어진 복사본(복도의 팔걸이의자 열 등)은 26/08/30부터 2D blob을 3D 연결성으로 다시
나눠(`--split-cell 0.10`, 셀 = 깊이별 샘플 간격×1.5) 별개 인스턴스가 된다. identity는 **8 m 이내 관측만**
으로 결정한다(`--near 8`): 멀리서 본 관측은 복사본들을 한 덩어리로 잇기 때문. 8 m 밖의 점은 identity 결정에는
안 쓰고, `gt_seg`/`segments.csv`에서는 **같은 색 객체 cloud 중 10 cm 이내로 가장 가까운 객체에 점 단위로 귀속**시킨다
(`--far-label 0.10`; 의자 열을 멀리서 봐도 의자별로 라벨, 근거리에서 한 번도 안 본 표면만 배경). `segments.csv`의
`is_far`=1 은 8 m 이내 점이 없는 관측. `gt_object_clouds`는 8 m 이내에서 본 표면만 담는다.

Identity 모델(색상 = prefab 타입, 인스턴스 = 3D voxel 겹침 분해; 거울/유리 유령 인스턴스 포함)과
산출물별 상세는 `meridian_benchmark/gt_build.py` docstring 참고. 시퀀스별 빌드 설정·통계는 각 GT 디렉터리의
`config.json`, 검증 결과는 `verify_report.json`에 기록된다.

## 벤치마크 실행

빌드/소싱 후 (`colcon build --packages-select meridian_benchmark && source install/setup.bash`):

```bash
# 터미널 1 — 하네스(player+recorder)만 뜬다. player는 테스트할 모듈이
# 구독을 붙일 때까지 발행을 미루고, 재생이 끝나면 launch 전체가 자동 종료된다.
# out 기본값은 bench_runs/<module>/<날짜_시간> — 이미 기록이 있는 디렉터리를
# 지정하면 recorder가 에러로 거부한다 (이전 실행과 섞인 채점 방지)
ros2 launch meridian_benchmark seg.launch.py \
  dataset:=~/yun/meridian_ws/datasets/uHumans2/apartment_scene/uHumans2_apartment_s1_00h \
  gt:=~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h  # end_frame:=50 부분 재생

# 터미널 2 — 테스트할 모듈을 직접 실행 (붙는 순간 재생 시작)
ros2 run meridian_seg seg_node                   # 예: upstream 스텁

# 채점 (run 디렉터리는 터미널 1 launch가 찍어준 timestamped 경로)
bench-score --module seg --gt ~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h \
  --run ~/yun/meridian_ws/bench_runs/seg/20260812_150000

# 5회 실행 + 중앙값 리포트 (plan #4; --out 아래에 <날짜_시간>/run_00..04 생성)
bench-run --module seg --runs 5 \
  --dataset ~/yun/meridian_ws/datasets/uHumans2/apartment_scene/uHumans2_apartment_s1_00h \
  --gt ~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h \
  --out ~/yun/meridian_ws/bench_runs/seg
```

launch 공통 인자: `dataset` `gt` `out` `input`(false = player 생략, 조합 실행용)
`module`(**기본 false** — true면 upstream 스텁 노드도 같이 띄움) `rate` `start_frame`
`end_frame` `pose_source`(base|cam) `pose_type`(plain=PoseStamped 기본 | cov).

| launch | player 주입 | 기록(채점) | 상태 |
|---|---|---|---|
| `seg` | rgb | segment_image → mask IoU, flowtime | 동작 |
| `clip` | rgb, seg | embedding set → flowtime, frames_dropped | 동작² |
| `geobuilder` | depth, info, seg, pose | instance_3d_set → voxel IoU, outlier, flowtime | 동작¹ |
| `geotracker` | instance3d, embedding | tracklet_set 수신 기록 | 주입 동작, 채점 TBD |
| `associator` | GT tracklet_set (1초 창, ~1 Hz) | decision_set·snapshot 수신 기록 | 주입 동작 (updater+graphcore 별도 기동 필요), 채점 TBD |
| `updater` / `graphcore` | — | 수신 기록만 | placeholder (plan TBD) |

¹ `/pose`는 `world_T_base`, 타입은 **PoseStamped** (SLAM 실제 출력 기준). 현재 geobuilder는
`PoseWithCovarianceStamped`를 구독하고 extrinsic 합성도 없으므로, upstream이 맞춰지기 전까지
geobuilder 단독 벤치는 `pose_type:=cov pose_source:=cam`으로 돌린다.

² clip 입력 구독은 **의도적으로 BEST_EFFORT** — 밀리면 drop하는 게 노드 설계라
`frames_dropped`가 flowtime과 함께 봐야 하는 일급 지표다. `module:=true`는
`clip_inference_node`를 띄우며, TensorRT engine 파일이 저장소에 있어야 기동한다
(모델 담당자가 git 추가 예정). slam 벤치는 제거됨 — upstream이 FAST-LIVO(3D LiDAR+IMU)로
전환되어 rgb-d 주입으로는 측정이 성립하지 않음 (docs/BENCHMARK_PLAN.md 참고).

조합 실행 예 (plan의 `input` 파라미터): `seg.launch.py` + `clip.launch.py input:=false`
— clip의 seg 입력은 GT가 아닌 seg 모듈 출력을 쓴다.

주의: upstream 각 노드는 SIGINT 종료 시 `rclpy.shutdown()`을 이중 호출해 traceback을 남기는데
(벤치 종료 시 항상 보임) 실행·기록·채점에는 영향이 없다.
