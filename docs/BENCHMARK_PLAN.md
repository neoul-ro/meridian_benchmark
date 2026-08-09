# Meridian Benchmark — 계획

데이터: uHumans2 apartment `00h` (1779 frames, 12.9 Hz), 경로 `~/yun/meridian_ws/datasets/uHumans2`.
GT identity는 `seg_cam` 색상 × 3D 인스턴스 분해로 유도 — **색상은 인스턴스가 아니라 prefab/재질 타입**이라
(동일 소품 여러 개가 한 색, 거울/유리 반사도 같은 색) 같은 색 관측을 3D 겹침으로 클러스터링해 인스턴스를 분리한다.
pose GT: **`/pose`는 `world_T_base`** (odom 200 Hz를 카메라 stamp로 보간) — slam ATE/RPE도 base 기준.
GT geometry 내부 계산(역투영·cloud)은 여기에 카메라 장착 변환(`tf_static`)을 곱한 `world_T_camera`를 쓰며,
`gt_poses.csv`에 base/cam 두 세트가 다 있다.
(위키 Data Types의 `/pose` 계약도 `world_T_base`로 갱신됨 — 26/08/08. geobuilder는 `base_T_camera`
extrinsic을 별도로 받아 합성.
방식: GT를 파일로 미리 만들어두고, 테스트할 모듈만 띄운 뒤 player가 입력 토픽을 GT 파일에서 발행 → 출력 기록 → 오프라인 채점.

## 테스트할 모듈과 metric

각각 모듈들은 다음처럼 실행됨.  
```
ros2 launch meridian_benchmark seg.launch.py
ros2 launch meridian_benchmark slam.launch.py
ros2 launch meridian_benchmark clip.launch.py
ros2 launch meridian_benchmark geobuilder.launch.py
ros2 launch meridian_benchmark geotracker.launch.py
ros2 launch meridian_benchmark associator.launch.py
ros2 launch meridian_benchmark updater.launch.py
ros2 launch meridian_benchmark graphcore.launch.py
```


| 모듈 | 주입 입력 | 출력 (채점 대상) | 채점 기준 GT | Metric | 비고 |
|---|---|---|---|---|---|
| `seg` | rgb | segment_image | GT segment_image | mask IoU | |
| `slam` | rgb, depth | pose | GT pose | ATE, RPE | |
| `clip` | rgb, segment_image | instance_embedding_set | — (시간만 측정) | flowtime | |
| `geobuilder` | depth, info, segment_image, pose | instance_3d_set | GT 객체 cloud | flowtime, voxel IoU, outlier 비율 | |
| `geotracker` | instance_3d_set, instance_embedding_set | tracklet_set | GT 객체 cloud (겹침으로 귀속) | tracklet purity, fragmentation | **TBD** |
| `associator` | (geotracker 출력) + graph_snapshot | association_decision_set | GT graph 상태 | MATCH precision/recall, false-merge/split | **TBD** |
| `updater` | (associator 출력) + graph_snapshot | object_update_set | GT graph 상태 | mutation 유효성, idempotency | **TBD** |
| `graphcore` | object_update_set | graph_snapshot, update_event | 자기 입력과 대조 (GT 불필요) | 불변식 통과 (version 단조, id 유일성, commit 일치) | **TBD** |

앞쪽(frontend) 모듈부터 진행한다. **TBD** 3개(geotracker+backend)는 frontend 평가가 자리잡은 뒤 확정. (일단 placeholder만))

모듈 조합 검증:

모듈 조합 검증용 통합 시험을 하기 위해 각 모듈에는 `input` parameter가 존재. 만약 seg/clip을 동시에 실행해서 조합성능을 본다면 rgb, depth는 gt로 publish되어야 하겠지만, clip의 입력인 seg 결과는 gt가 아닌 seg module로 출력이 될 것이므로,
```
ros2 launch meridian_benchmark seg.launch.py
ros2 launch meridian_benchmark clip.launch.py input:=false
```


## 생성할 GT 파일

위 표의 "주입 입력"과 "채점 기준 GT" 열이 곧 생성 목록이다:

- **주입용** — 프레임별: segment_image(색상→uint8 remap), pose, embedding(객체당 직교 벡터), instance cloud
- **채점용** — segment↔gt_object 대응표, 객체별 누적 cloud/centroid/AABB
- **TBD (backend용)** — 프레임별 graph 상태 (associator/updater 주입·채점 겸용)

## 정해둔 것

1. GT 파일은 시퀀스당 1회 생성, player가 profile별로 필요한 토픽만 골라 발행
2. `/tracklet_set`, `/association_decision_set`은 GT 정의 불가 → 만들지 않고 채점 때 GT 참조
   (그래서 associator/updater는 실제 상류 출력을 입력으로 받음)
3. upstream 저장소는 수정하지 않음 (자체 launch만)
4. 실시간 재생, 5회 실행 중앙값 리포트
5. player는 구독자가 다 붙을 때까지 발행을 미룸 (upstream QoS가 `RELIABLE KEEP_LAST depth=10`이라
   먼저 쏘면 앞 프레임 유실)
6. flowtime은 recorder가 입·출력 토픽 수신 시각 차로 외부 측정 → 내부 처리시간의 **상한**.
   clip/geobuilder 대표치는 경합 없는 모듈별 격리 실행에서 뽑음
7. TODO: dynamic 객체(사람 등장 시퀀스), L2 profile

## 순서

1. GT 파일 생성
2. player(GT 발행) + recorder
3. 채점기 (위 metric)
4. 모듈별 실행 + L1 baseline

