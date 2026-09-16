# Perception Frontend -> DA boundary. One message per keyframe-gate fire = one frame snapshot.
# Every array is aligned to tracklet_ids order. points: world xyz, tracklet i owns the next points_num[i] points.
# id_emb / query_emb: 512 floats per tracklet, L2-normalized. All-zero row = no embedding in this message.

std_msgs/Header header          # stamp = camera capture time, frame_id = 'world'
int64[] tracklet_ids            # (N)
uint32[] points_num             # (N)
float32[] points                # (3 * sum(points_num))
float32[] depth_valid_ratio     # (N) fraction of the mask that carried usable depth
float32[] seg_conf              # (N) segmentor confidence
float32[] id_emb                # (512 * N) identity vector (mask-weighted patch pool)
float32[] query_emb             # (512 * N) open-vocabulary query vector (CLS)
