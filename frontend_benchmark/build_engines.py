#!/usr/bin/env python3
"""frontend 가 쓰는 TensorRT 엔진 2개를 이 PC(x86 + RTX 3060)용으로 다시 빌드한다.

받은 src/meridian_frontend/engines/*.plan 은 Jetson(Orin) 플랫폼 태그라 x86 TensorRT 가 거부한다
("Platform specific tag mismatch"). 원본은 건드리지 않고 models/frontend_rtx3060/ 에 새로 만든다.

  fastsam.plan     ← models/fastsam/FastSAM-s-1024.onnx  (I/O 이름·shape 가 sam.py 바인딩과 동일)
  clip_image.plan  ← models/clip/clip_vit_b32_visual_pooled.onnx
                     I/O 이름만 clip.py 바인딩에 맞춘다:
                       images         → pixel_values  (m,3,224,224)
                       patch_weights  → wpatch        (m,49)
                       embeddings     → id_emb        (mask-weighted patch pool — patch_weights 를 쓰는 출력)
                       cls_embeddings → query_emb     (CLS)
                     동적 배치 1..128 (sam.LANES).

둘 다 FP16 (받은 엔진의 커널 이름이 f16 — 같은 정밀도).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import WS, MODELS  # noqa: E402

CLIP_RENAME = {'images': 'pixel_values', 'patch_weights': 'wpatch',
               'embeddings': 'id_emb', 'cls_embeddings': 'query_emb'}


def build(onnx_path, out_path, rename=None, profile=None, log=print):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            log(f'  parse error: {parser.get_error(i)}')
        raise SystemExit(f'ONNX 파싱 실패: {onnx_path}')
    if rename:
        tensors = [network.get_input(i) for i in range(network.num_inputs)] + \
                  [network.get_output(i) for i in range(network.num_outputs)]
        for t in tensors:
            if t.name in rename:
                t.name = rename[t.name]
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if profile:
        p = builder.create_optimization_profile()
        for name, (mn, opt, mx) in profile.items():
            p.set_shape(name, mn, opt, mx)
        config.add_optimization_profile(p)
    io = []
    for i in range(network.num_inputs):
        t = network.get_input(i); io.append(('IN', t.name, tuple(t.shape)))
    for i in range(network.num_outputs):
        t = network.get_output(i); io.append(('OUT', t.name, tuple(t.shape)))
    for row in io:
        log(f'  {row[0]:3s} {row[1]} {row[2]}')
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit(f'엔진 빌드 실패: {onnx_path}')
    out_path.write_bytes(bytes(blob))
    log(f'  → {out_path} ({out_path.stat().st_size / 2**20:.1f} MB, {time.time() - t0:.0f}s)')
    return io


def verify(path, expect, log=print):
    """다시 읽어서 frontend 가 바인딩하는 이름이 전부 있는지 확인."""
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = rt.deserialize_cuda_engine(path.read_bytes())
    if eng is None:
        raise SystemExit(f'재로드 실패: {path}')
    names = {eng.get_tensor_name(i): (str(eng.get_tensor_mode(eng.get_tensor_name(i))),
                                      tuple(eng.get_tensor_shape(eng.get_tensor_name(i))))
             for i in range(eng.num_io_tensors)}
    missing = [n for n in expect if n not in names]
    if missing:
        raise SystemExit(f'{path.name}: frontend 가 바인딩하는 텐서 없음 {missing} (있는 것 {list(names)})')
    log(f'  재로드 OK {path.name}: {names}')
    return names


def sha1(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(MODELS / 'frontend_rtx3060'))
    ap.add_argument('--only', choices=['seg', 'clip'])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {'tensorrt': trt.__version__, 'precision': 'fp16', 'built_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    if a.only in (None, 'seg'):
        src = MODELS / 'fastsam/FastSAM-s-1024.onnx'
        print(f'[seg] {src}', flush=True)
        build(src, out / 'fastsam.plan')
        verify(out / 'fastsam.plan', ['images', 'output0', 'output1'])
        manifest['fastsam.plan'] = {'onnx': str(src), 'onnx_sha1': sha1(src)}
    if a.only in (None, 'clip'):
        src = MODELS / 'clip/clip_vit_b32_visual_pooled.onnx'
        print(f'[clip] {src}', flush=True)
        prof = {'pixel_values': ((1, 3, 224, 224), (32, 3, 224, 224), (128, 3, 224, 224)),
                'wpatch': ((1, 49), (32, 49), (128, 49))}
        build(src, out / 'clip_image.plan', rename=CLIP_RENAME, profile=prof)
        verify(out / 'clip_image.plan', ['pixel_values', 'wpatch', 'query_emb', 'id_emb'])
        manifest['clip_image.plan'] = {'onnx': str(src), 'onnx_sha1': sha1(src), 'rename': CLIP_RENAME,
                                       'batch_profile': [1, 32, 128]}
    mf = out / 'manifest.json'
    old = json.loads(mf.read_text()) if mf.exists() else {}
    old.update(manifest)
    mf.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    print('done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
