# Meridian Benchmark — 계획

데이터: uHumans2 apartment `00h` (1779 frames, 12.9 Hz), 경로 `~/yun/meridian_ws/datasets/uHumans2`.
GT identity는 `seg_cam` 색상 × 3D 인스턴스 분해로 유도 — **색상은 인스턴스가 아니라 prefab/재질 타입**이라
(동일 소품 여러 개가 한 색, 거울/유리 반사도 같은 색) 같은 색 관측을 3D 겹침으로 클러스터링해 인스턴스를 분리한다
(프레임 내 2D blob도 3D 연결성으로 먼저 나눔 — 화면에서만 겹치는 복사본은 별개, 실제 접촉한 것만 병합; identity는 8 m 이내 관측으로만 결정, 8 m 밖 픽셀은 가장 가까운 객체에 귀속).
pose GT: **`/pose`는 `world_T_base`** (odom 200 Hz를 카메라 stamp로 보간).
GT geometry 내부 계산(역투영·cloud)은 여기에 카메라 장착 변환(`tf_static`)을 곱한 `world_T_camera`를 쓰며,
`gt_poses.csv`에 base/cam 두 세트가 다 있다.
(위키 Data Types의 `/pose` 계약도 `world_T_base`로 갱신됨 — 26/08/08. geobuilder는 `base_T_camera`
extrinsic을 별도로 받아 합성.
방식: GT를 파일로 미리 만들어두고, 테스트할 모듈만 띄운 뒤 player가 입력 토픽을 GT 파일에서 발행 → 출력 기록 → 오프라인 채점.

## 테스트할 모듈과 metric

각각 모듈들은 다음처럼 실행됨.  
```
ros2 launch meridian_benchmark seg.launch.py
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
| `clip` | rgb, segment_image | instance_embedding_set | — (시간만 측정) | flowtime, frames_dropped | 입력 구독이 의도적으로 BEST_EFFORT → drop이 정상 지표 |
| `geobuilder` | depth, info, segment_image, pose | instance_3d_set | GT 객체 cloud | flowtime, voxel IoU, outlier 비율 | |
| `geotracker` | instance_3d_set, instance_embedding_set | tracklet_set | GT 객체 cloud (겹침으로 귀속) | tracklet purity, fragmentation | **TBD** |
| `associator` | **GT tracklet (가시 구간 단위, 토픽 보류)** + graph_snapshot | association_decision_set | gt_tracklets_index.csv (tracklet→gt_object 대응표) | MATCH precision/recall, false-merge/split | 채점기 TBD |
| `updater` | (associator 출력) + graph_snapshot | object_update_set | GT graph 상태 | mutation 유효성, idempotency | **TBD** |
| `graphcore` | object_update_set | graph_snapshot, update_event | 자기 입력과 대조 (GT 불필요) | 불변식 통과 (version 단조, id 유일성, commit 일치) | **TBD** |

앞쪽(frontend) 모듈부터 진행한다. **TBD** 3개(geotracker+backend)는 frontend 평가가 자리잡은 뒤 확정. (일단 placeholder만))

**slam 벤치는 제거** (26/08/12): upstream이 FAST-LIVO(3D LiDAR+IMU 기반)로 전환되어 rgb-d 주입으로는
측정 자체가 성립하지 않고, 굳이 필요 없다고 판단. upstream 참고 사항 — FAST-LIVO의 `/pose`는
발행 시점 `now()` 스탬프이고 `world_T_imu`라 위키 계약(`world_T_base`, capture-time stamp)과 다름.

모듈 조합 검증:

모듈 조합 검증용 통합 시험을 하기 위해 각 모듈에는 `input` parameter가 존재. 만약 seg/clip을 동시에 실행해서 조합성능을 본다면 rgb, depth는 gt로 publish되어야 하겠지만, clip의 입력인 seg 결과는 gt가 아닌 seg module로 출력이 될 것이므로,
```
ros2 launch meridian_benchmark seg.launch.py
ros2 launch meridian_benchmark clip.launch.py input:=false
```


## 생성할 GT 파일

위 표의 "주입 입력"과 "채점 기준 GT" 열이 곧 생성 목록이다:

- **주입용** — 프레임별: segment_image(색상→uint8 remap), pose, embedding(객체당 직교 벡터), instance cloud
- **주입용 (DA)** — `gt_tracklets.h5`: GT tracklet = 객체의 **연속 가시 구간**(관측 공백 ≤5프레임 연결),
  시야에서 사라지는 프레임에 발행, id는 발행 순 1씩 증가. 점군은 그 구간에 카메라가 본 표면만
  2 cm map-frame 복셀(카메라에서 5 m 초과인 점은 버림), semantics는 upstream clip 노드 파이프라인으로 계산.
  한 축이라도 extent ≤4 cm인 tracklet(복셀 1~2층)과 천장 30 cm 띠 안에 완전히 든 tracklet은 제외. 카메라 거리(min/mean)를 메타데이터로 기록. 파일 계층 = `meridian_msgs/Tracklet` + `_metadata/` (26/08/29 확정; 발행 토픽/타입은 보류)
- **채점용** — segment↔gt_object 대응표, 객체별 누적 cloud/centroid/AABB, `gt_tracklets_index.csv`
  (tracklet_id→gt_object_id)
- **TBD (backend용)** — 프레임별 graph 상태 (updater 주입·채점용)

## 정해둔 것

1. GT 파일은 시퀀스당 1회 생성, player가 profile별로 필요한 토픽만 골라 발행
2. tracklet GT는 알고리즘이 아니라 **규약으로 정의**한다: 객체가 시야에서 사라질 때 그 가시 구간을
   tracklet 하나로 (26/08/29 확정, 이전 고정 1초 창 방식 폐기). "단일 정답 없음" 문제를 정책 고정으로 해소.
   `/association_decision_set`은 여전히 GT를 만들지 않음 (updater는 실제 associator 출력을 받음)
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

