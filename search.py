"""설정 고르기 — 테스트를 보지 않고. `python search.py`

과제 규칙은 "하이퍼파라미터는 모든 fold에서 같게 두고, 테스트 사람의 데이터는
하이퍼파라미터 선택에 쓰지 않는다"이다. 그래서 이렇게 한다.

1. 개발용 fold 하나(기본 p00)만 쓴다.
2. 그 fold의 **안쪽 검증 2명**(p01, p02) 오차로만 설정을 고른다.
3. `track_test=False`로 테스트 오차를 **계산조차 하지 않는다.** 눈에 띌 일이 없다.
4. 고른 설정을 15 fold 전부에 그대로 적용한다.

남는 한계는 분명히 적어 둔다. 검증을 맡은 p01·p02는 다른 fold에서는 테스트
대상이므로, 그 두 fold에 한해 "테스트 사람의 데이터가 설정 선택에 쓰였다"에
걸린다. 사람 15명으로 LOPO를 하면서 이를 완전히 피하려면 설정 선택 전용으로
사람을 따로 떼어 두어야 하고, 그만큼 학습 데이터가 줄어든다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

if "--device" in sys.argv:   # common.backend가 import 시점에 읽는다
    os.environ["GAZE_DEVICE"] = sys.argv[sys.argv.index("--device") + 1]

from train import Config, THREADS_PER_FOLD, train_fold

# 1차 학습에서 본 증상: 학습 손실 0.35 → 0.046인데 검증은 epoch 1부터 평평했다.
# 사람 외모에 과적합한다. 후보는 전부 "덜 외우게 만드는" 쪽이다.
REGULARIZATION = {
    "base":                        dict(dropout=0.2, weight_decay=0.0, augment=False),
    "dropout 0.5":                 dict(dropout=0.5, weight_decay=0.0, augment=False),
    "wd 1e-4":                     dict(dropout=0.2, weight_decay=1e-4, augment=False),
    "augment":                     dict(dropout=0.2, weight_decay=0.0, augment=True),
    "augment + dropout 0.5":       dict(dropout=0.5, weight_decay=0.0, augment=True),
    "augment + wd 1e-4":           dict(dropout=0.2, weight_decay=1e-4, augment=True),
    "augment + do 0.35 + wd 3e-4": dict(dropout=0.35, weight_decay=3e-4, augment=True),
    "augment + do 0.5 + wd 1e-4":  dict(dropout=0.5, weight_decay=1e-4, augment=True),
    "augment + do 0.6 + wd 1e-3":  dict(dropout=0.6, weight_decay=1e-3, augment=True),
    "augment + do 0.5 + wd 1e-3":  dict(dropout=0.5, weight_decay=1e-3, augment=True),
}

# 정규화를 고른 뒤 용량을 본다. 과적합이 심했으니 키우는 쪽이 꼭 좋다는 보장이 없고,
# 그래서 재 본다. REG는 --reg로 넣는다(기본값은 1단계 1위).
CAPACITY = {
    "narrow (16,32,64) h128":   dict(conv_channels=(16, 32, 64), hidden=128, joint=64),
    "base   (24,48,96) h256":   dict(conv_channels=(24, 48, 96), hidden=256, joint=96),
    "wide   (32,64,128) h256":  dict(conv_channels=(32, 64, 128), hidden=256, joint=96),
    "wide   (32,64,128) h512":  dict(conv_channels=(32, 64, 128), hidden=512, joint=128),
    "deep   (24,48,96,96) h256": dict(conv_channels=(24, 48, 96, 96), hidden=256, joint=96),
}

PRESETS = {"regularization": REGULARIZATION, "capacity": CAPACITY}


def _worker(args):
    name, cfg, person, log_dir = args
    r = train_fold(cfg, person, log_dir=log_dir, track_test=False, verbose=False)
    best = min(r["history"], key=lambda h: h["val"])
    print(f"  {name:<30} 검증 {best['val']:.3f}°  (epoch {best['epoch']}, "
          f"학습 손실 {best['loss']:.4f}, {r['seconds'] / 60:.1f}분)", flush=True)
    return {"name": name, "val": best["val"], "epoch": best["epoch"],
            "history": r["history"], "config": {k: v for k, v in cfg.__dict__.items()}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="p00", help="개발용 fold. 이 fold의 테스트는 보지 않는다")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--preset", choices=sorted(PRESETS), default="regularization")
    p.add_argument("--reg", nargs="*", default=[],
                   help="capacity 단계에서 고정할 정규화 설정, 예: dropout=0.5 weight_decay=1e-4 augment=1")
    p.add_argument("--jobs", type=int, default=0, help="0이면 후보 수만큼 한꺼번에 (GPU는 1 권장)")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    candidates = dict(PRESETS[args.preset])
    if args.reg:
        fixed = {}
        for item in args.reg:
            key, _, value = item.partition("=")
            if key not in Config.__dataclass_fields__:
                raise SystemExit(f"Config에 없는 항목이다: {key}")
            fixed[key] = bool(int(value)) if key == "augment" else float(value)
        candidates = {name: {**kwargs, **fixed} for name, kwargs in candidates.items()}
        print(f"모든 후보에 고정한 설정: {fixed}")

    args.out = args.out or f"results/search-{args.preset}"
    os.makedirs(args.out, exist_ok=True)
    def slug(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z.]+", "-", name).strip("-")

    jobs = [(name, Config(epochs=args.epochs, **kwargs), args.fold,
             os.path.join(args.out, slug(name)))
            for name, kwargs in candidates.items()]

    from common.backend import describe
    workers = args.jobs or len(jobs)
    print(f"개발용 fold {args.fold} · 후보 {len(jobs)}개 · {args.epochs} epoch · "
          f"worker {workers}개 · {describe()}")
    print("테스트 오차는 계산하지 않는다 (track_test=False)\n")

    t0 = time.perf_counter()
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    print(f"\n{(time.perf_counter() - t0) / 60:.1f}분")

    results.sort(key=lambda r: r["val"])
    print(f"\n{'순위':<4}{'설정':<32}{'검증 오차':>10}{'epoch':>7}")
    for i, r in enumerate(results, 1):
        print(f"{i:<4}{r['name']:<32}{r['val']:>9.3f}°{r['epoch']:>7}")

    base = next((r for r in results if r["name"].strip().startswith("base")), results[-1])
    win = results[0]
    print(f"\n고른 설정: {win['name']}  "
          f"(기준 {base['name']} {base['val']:.3f}° → {win['val']:.3f}°, "
          f"{win['val'] - base['val']:+.3f}°)")

    path = os.path.join(args.out, "search.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
