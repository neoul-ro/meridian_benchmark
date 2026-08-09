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

# 검증 (A 구조 / B 일관성 / C 자기 재투영 / D 프레임 간 3D 일관성 / E 인스턴스 분리)
python3 -m meridian_benchmark.gt_verify \
  --dataset ~/yun/meridian_ws/datasets/uHumans2/apartment_scene/uHumans2_apartment_s1_00h \
  --gt ~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h
```

Identity 모델(색상 = prefab 타입, 인스턴스 = 3D voxel 겹침 분해; 거울/유리 유령 인스턴스 포함)과
산출물별 상세는 `meridian_benchmark/gt_build.py` docstring 참고. 시퀀스별 빌드 설정·통계는 각 GT 디렉터리의
`config.json`, 검증 결과는 `verify_report.json`에 기록된다.

## 벤치마크 실행

빌드/소싱 후 (`colcon build --packages-select meridian_benchmark && source install/setup.bash`):

```bash
# 터미널 1 — 하네스(player+recorder)만 뜬다. player는 테스트할 모듈이
# 구독을 붙일 때까지 발행을 미루고, 재생이 끝나면 launch 전체가 자동 종료된다
ros2 launch meridian_benchmark seg.launch.py \
  dataset:=~/yun/meridian_ws/datasets/uHumans2/apartment_scene/uHumans2_apartment_s1_00h \
  gt:=~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h \
  out:=~/yun/meridian_ws/bench_runs/seg/run0     # end_frame:=50 으로 부분 재생

# 터미널 2 — 테스트할 모듈을 직접 실행 (붙는 순간 재생 시작)
ros2 run meridian_seg seg_node                   # 예: upstream 스텁

# 채점
bench-score --module seg --gt ~/yun/meridian_ws/datasets/gt/uHumans2_apartment_s1_00h \
  --run ~/yun/meridian_ws/bench_runs/seg/run0

# 5회 실행 + 중앙값 리포트 (plan #4)
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
| `slam` | rgb, depth, info | /pose (PoseStamped) → ATE/RPE | 하네스만 (rgb-d SLAM 실행체 없음, 모듈은 별도 기동) |
| `clip` | rgb, seg | embedding set → flowtime | 동작 |
| `geobuilder` | depth, info, seg, pose | instance_3d_set → voxel IoU, outlier, flowtime | 동작¹ |
| `geotracker` | instance3d, embedding | tracklet_set 수신 기록 | 주입 동작, 채점 TBD |
| `associator` / `updater` / `graphcore` | — | 수신 기록만 | placeholder (plan TBD) |

¹ `/pose`는 `world_T_base`, 타입은 **PoseStamped** (SLAM 실제 출력 기준). 현재 geobuilder는
`PoseWithCovarianceStamped`를 구독하고 extrinsic 합성도 없으므로, upstream이 맞춰지기 전까지
geobuilder 단독 벤치는 `pose_type:=cov pose_source:=cam`으로 돌린다.

조합 실행 예 (plan의 `input` 파라미터): `seg.launch.py` + `clip.launch.py input:=false`
— clip의 seg 입력은 GT가 아닌 seg 모듈 출력을 쓴다.

주의: upstream 각 노드는 SIGINT 종료 시 `rclpy.shutdown()`을 이중 호출해 traceback을 남기는데
(벤치 종료 시 항상 보임) 실행·기록·채점에는 영향이 없다.
