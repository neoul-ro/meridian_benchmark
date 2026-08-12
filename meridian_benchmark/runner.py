"""Benchmark runner: N real-time replays of one module, median report.

Wraps `ros2 launch meridian_benchmark <module>.launch.py` (plan #4: 5 runs,
median). Requires a sourced workspace (ros2 on PATH, meridian_msgs built).

  bench-run --module seg --dataset <seq dir> --gt <gt dir> \
            --out ~/yun/meridian_ws/bench_runs/seg [--runs 5] \
            [--launch-arg rate:=1.0 ...]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .score import SCORERS, score_run


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--module', required=True, choices=sorted(SCORERS))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True, help='report + per-run dirs')
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--launch-arg', action='append', default=[],
                    help='extra key:=value forwarded to ros2 launch')
    a = ap.parse_args()

    # timestamped session dir: re-running with the same --out never
    # collides with (or silently re-scores) an earlier session
    out = Path(a.out).expanduser() / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    for k in range(a.runs):
        run_dir = out / f'run_{k:02d}'
        cmd = ['ros2', 'launch', 'meridian_benchmark',
               f'{a.module}.launch.py', f'dataset:={a.dataset}',
               f'gt:={a.gt}', f'out:={run_dir}'] + a.launch_arg
        print(f'[bench-run] run {k + 1}/{a.runs}: {" ".join(cmd)}',
              flush=True)
        t0 = time.monotonic()
        proc = subprocess.run(cmd)
        print(f'[bench-run] run {k + 1} exited {proc.returncode} '
              f'({time.monotonic() - t0:.0f}s)', flush=True)
        s = score_run(a.module, a.gt, str(run_dir))
        with open(run_dir / 'score.json', 'w') as f:
            json.dump(s, f, indent=1)
        scores.append(s)

    flat = [_flatten(s) for s in scores]
    keys = sorted(set().union(*[set(f) for f in flat])) if flat else []
    median = {k: float(np.median([f[k] for f in flat if k in f]))
              for k in keys}
    report = {'module': a.module, 'dataset': a.dataset, 'gt': a.gt,
              'runs': len(scores), 'median': median, 'per_run': scores}
    with open(out / 'report.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps({'median': median}, indent=1))
    print(f'-> {out / "report.json"}')


if __name__ == '__main__':
    main()
