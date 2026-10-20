"""A8 — 해석용 eval-only. 학습하지 않고 저장된 가중치로만 돌린다.

  1) occlusion sensitivity: 입력의 한 조각을 가리고 오차가 얼마나 오르나
  2) head pose 제거/섞기: 입력을 망가뜨리면 오차가 얼마나 오르나
  3) person-ID linear probe: penultimate 특징에서 사람을 얼마나 맞히나

`python interpret.py --device gpu --run results/p1_seed0`
"""
from __future__ import annotations

import argparse
import json
import os
import sys

if "--device" in sys.argv:
    os.environ["GAZE_DEVICE"] = sys.argv[sys.argv.index("--device") + 1]

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run", default="results/p1_seed0")
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=1500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(a.run, "interpret.npz")

    from common.backend import describe, to_cpu, to_device, xp
    if xp.__name__ == "cupy":
        free, total = xp.cuda.runtime.memGetInfo()
        xp.get_default_memory_pool().set_limit(size=int(min(total * 0.30, free * 0.7)))
    print(describe(), flush=True)

    import dataset
    from dataset import ImageNorm, PERSONS, Standardizer
    from metrics import angular_error
    from model import GazeNet
    from train import split_persons

    samples = dataset.load(verbose=False)
    cfg = json.load(open(os.path.join(a.run, "lopo.json"), encoding="utf-8"))["config"]
    H, W = 36, 60
    rng = np.random.default_rng(0)

    occl = np.zeros((len(PERSONS), (H - a.patch) // a.stride + 1,
                     (W - a.patch) // a.stride + 1), dtype=np.float32)
    rows_head, rows_probe = [], []

    for pi, person in enumerate(PERSONS):
        fold_npz = os.path.join(a.run, "folds", f"{person}.npz")
        if not os.path.exists(fold_npz):
            print(f"  {person}: 가중치 없음, 건너뜀"); continue
        tr_p, _ = split_persons(person, cfg["n_val_persons"], list(PERSONS))
        tr = samples.only(*tr_p)
        norms = (ImageNorm.fit(tr.image), Standardizer.fit(tr.head),
                 Standardizer.fit(tr.gaze))
        image_norm, head_std, gaze_std = norms

        net = GazeNet(conv_channels=tuple(cfg["conv_channels"]), hidden=cfg["hidden"],
                      joint=cfg["joint"], dropout=cfg["dropout"], norm=cfg["norm"],
                      kernel=cfg.get("kernel", 3), n_pools=cfg.get("n_pools"),
                      use_head=cfg.get("use_head", True), loss="mse",
                      gaze_mean=gaze_std.mean, gaze_std=gaze_std.std, seed=cfg["seed"])
        with np.load(fold_npz) as z:
            net.load_state_dict({k[2:]: z[k] for k in z.files if k.startswith("w.")})

        te = samples.only(person)
        n = min(a.max_samples, len(te))
        idx = rng.choice(len(te), size=n, replace=False)
        img = to_device(te.image[idx])
        hd = head_std.transform(to_device(te.head[idx]))
        gz = te.gaze[idx]

        def err_of(image_dev, head_dev):
            pr = to_cpu(net.predict(image_norm(image_dev), head_dev, train_flg=False))
            return float(angular_error(gaze_std.inverse(pr), gz).mean())

        base = err_of(img, hd)

        # 1) occlusion
        mean_val = xp.asarray(np.float32(127.5))
        for iy, y in enumerate(range(0, H - a.patch + 1, a.stride)):
            for ix, x in enumerate(range(0, W - a.patch + 1, a.stride)):
                blocked = img.astype(xp.float32).copy()
                blocked[:, :, y:y + a.patch, x:x + a.patch] = mean_val
                occl[pi, iy, ix] = err_of(blocked, hd) - base

        # 2) head pose 망가뜨리기
        zero = err_of(img, xp.zeros_like(hd))
        perm = err_of(img, hd[to_device(rng.permutation(n))])
        rows_head.append([person, round(base, 4), round(zero - base, 4),
                          round(perm - base, 4)])

        # 3) person-ID linear probe (학습 사람들의 특징으로 로지스틱 회귀)
        with np.load(fold_npz) as z:
            if "feat_train" in z.files:
                F, y = z["feat_train"].astype(np.float64), z["feat_train_person"]
                F = (F - F.mean(0)) / (F.std(0) + 1e-6)
                ntr = int(0.7 * len(F))
                sh = rng.permutation(len(F))
                tri, tei = sh[:ntr], sh[ntr:]
                K = int(y.max()) + 1
                Wm = np.zeros((F.shape[1], K)); bm = np.zeros(K)
                lr_, Y = 0.5, np.eye(K)[y[tri]]
                for _ in range(300):
                    logit = F[tri] @ Wm + bm
                    logit -= logit.max(1, keepdims=True)
                    P = np.exp(logit); P /= P.sum(1, keepdims=True)
                    G = (P - Y) / len(tri)
                    Wm -= lr_ * (F[tri].T @ G + 1e-4 * Wm); bm -= lr_ * G.sum(0)
                acc = float((np.argmax(F[tei] @ Wm + bm, 1) == y[tei]).mean())
                rows_probe.append([person, K, round(acc, 4), round(1 / K, 4)])
        print(f"  {person}: 기준 {base:.3f}°  head0 {zero - base:+.3f}  "
              f"head섞기 {perm - base:+.3f}  "
              f"probe {rows_probe[-1][2] if rows_probe else float('nan'):.3f}", flush=True)

    np.savez_compressed(out_path, occlusion=occl, persons=np.array(PERSONS),
                        head_rows=np.array(rows_head, dtype=object),
                        probe_rows=np.array(rows_probe, dtype=object),
                        patch=a.patch, stride=a.stride)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
