"""B0 — 처리량 격자. batch × per-gpu. `python bench_throughput.py --device gpu`

PREREG 1절. 개발용 fold 1 epoch를 돌려 samples/s와 epoch 시간을 잰다.
per-gpu는 같은 카드에 프로세스를 그만큼 띄워 합계 처리량을 본다.
nvidia-smi를 1초 간격으로 따로 읽어 사용률·메모리를 평균한다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

if "--device" in sys.argv:
    os.environ["GAZE_DEVICE"] = sys.argv[sys.argv.index("--device") + 1]

import numpy as np


class SmiSampler(threading.Thread):
    """nvidia-smi를 1초 간격으로 읽어 이 프로세스가 쓰는 카드의 사용률·메모리를 모은다."""

    def __init__(self, index: int):
        super().__init__(daemon=True)
        self.index = index
        self.util, self.mem = [], []
        self._done = threading.Event()  # Thread._stop을 가리면 안 된다

    def run(self):
        q = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", f"--id={self.index}"]
        while not self._done.is_set():
            try:
                out = subprocess.run(q, capture_output=True, text=True, timeout=5).stdout
                u, m = out.strip().split(", ")
                self.util.append(float(u)); self.mem.append(float(m))
            except Exception:
                pass
            self._done.wait(1.0)

    def stop(self):
        self._done.set()
        self.join(timeout=3)
        return (float(np.mean(self.util)) if self.util else float("nan"),
                float(np.mean(self.mem)) if self.mem else float("nan"))


def _one(args):
    """프로세스 하나가 1 epoch 돌고 (samples, seconds)를 돌려준다."""
    batch, fold, cap_frac = args
    from common.backend import on_gpu, synchronize, xp
    if on_gpu():
        free, total = xp.cuda.runtime.memGetInfo()
        xp.get_default_memory_pool().set_limit(size=int(min(total * cap_frac, free * 0.7)))

    import dataset
    from common.optimizer import Adam
    from dataset import Batches, ImageNorm, PERSONS, Standardizer
    from model import GazeNet
    from train import cosine_lr, split_persons

    all_s = dataset.load(verbose=False)
    train_p, _ = split_persons(fold, 2, list(PERSONS))
    tr = all_s.only(*train_p)
    norms = (ImageNorm.fit(tr.image), Standardizer.fit(tr.head), Standardizer.fit(tr.gaze))
    trd = tr.to_device()

    net = GazeNet(dropout=0.5, seed=0)
    lr = 2e-3 * np.sqrt(batch / 128)
    opt = Adam(lr=lr)
    bl = Batches(trd, batch, *norms, shuffle=True, drop_last=True, seed=0)

    # 워밍업 몇 스텝은 빼고 잰다 (첫 커널 컴파일·할당)
    it = iter(bl)
    for _ in range(3):
        img, hd, gz, _ = next(it)
        opt.update(net.params, net.gradient(img, hd, gz))
    synchronize()

    t0 = time.perf_counter()
    steps = 0
    for img, hd, gz, _ in bl:
        opt.lr = cosine_lr(steps, len(bl), lr, 0.02)
        opt.update(net.params, net.gradient(img, hd, gz))
        steps += 1
    synchronize()
    return steps * batch, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fold", default="p00")
    ap.add_argument("--batches", type=int, nargs="*", default=[128, 256, 512, 1024])
    ap.add_argument("--per-gpu", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--cap-frac", type=float, default=0.30)
    ap.add_argument("--out", default="results/bench/b0_throughput.json")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    from multiprocessing import get_context

    rows = []
    print(f"{'batch':>7}{'per-gpu':>9}{'samples/s':>12}{'epoch s':>10}"
          f"{'카드당 배수':>12}{'util%':>8}{'mem MiB':>10}")
    base = {}
    for batch in a.batches:
        for k in a.per_gpu:
            smi = SmiSampler(0); smi.start()
            t0 = time.perf_counter()
            try:
                with get_context("spawn").Pool(k) as pool:
                    out = pool.map(_one, [(batch, a.fold, a.cap_frac / max(k, 1))] * k)
            except Exception as exc:
                smi.stop()
                print(f"{batch:>7}{k:>9}   실패: {type(exc).__name__}")
                rows.append({"batch": batch, "per_gpu": k, "error": repr(exc)[:200]})
                continue
            wall = time.perf_counter() - t0
            util, mem = smi.stop()

            total_samples = sum(s for s, _ in out)
            worker_s = float(np.mean([sec for _, sec in out]))
            rate = total_samples / max(np.max([sec for _, sec in out]), 1e-9)
            base.setdefault(batch, rate)
            rows.append({"batch": batch, "per_gpu": k, "samples_per_s": rate,
                         "epoch_s_per_worker": worker_s, "wall_s": wall,
                         "gain_vs_per_gpu1": rate / base[batch],
                         "util_pct": util, "mem_mib": mem})
            print(f"{batch:>7}{k:>9}{rate:>12,.0f}{worker_s:>10.1f}"
                  f"{rate / base[batch]:>12.2f}{util:>8.0f}{mem:>10.0f}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
