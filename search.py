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
import time

from train import Config, THREADS_PER_FOLD, train_fold

# 1차 학습에서 본 증상: 학습 손실 0.35 → 0.046인데 검증은 epoch 1부터 평평했다.
# 사람 외모에 과적합한다. 후보는 전부 "덜 외우게 만드는" 쪽이다.
CANDIDATES = {
    "base (현행)":               dict(dropout=0.2, weight_decay=0.0, augment=False),
    "dropout 0.5":               dict(dropout=0.5, weight_decay=0.0, augment=False),
    "wd 1e-4":                   dict(dropout=0.2, weight_decay=1e-4, augment=False),
    "augment":                   dict(dropout=0.2, weight_decay=0.0, augment=True),
    "augment + dropout 0.5":     dict(dropout=0.5, weight_decay=0.0, augment=True),
    "augment + wd 1e-4":         dict(dropout=0.2, weight_decay=1e-4, augment=True),
    "augment + do 0.35 + wd 3e-4": dict(dropout=0.35, weight_decay=3e-4, augment=True),
    "augment + do 0.5 + wd 1e-4": dict(dropout=0.5, weight_decay=1e-4, augment=True),
}


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
    p.add_argument("--out", default="results/search")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = [(name, Config(epochs=args.epochs, **kwargs), args.fold,
             os.path.join(args.out, name.replace(" ", "_").replace("+", "")))
            for name, kwargs in CANDIDATES.items()]

    print(f"개발용 fold {args.fold} · 후보 {len(jobs)}개 · {args.epochs} epoch · "
          f"worker당 BLAS 스레드 {THREADS_PER_FOLD}개")
    print("테스트 오차는 계산하지 않는다 (track_test=False)\n")

    t0 = time.perf_counter()
    from multiprocessing import get_context
    with get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)
    print(f"\n{(time.perf_counter() - t0) / 60:.1f}분")

    results.sort(key=lambda r: r["val"])
    print(f"\n{'순위':<4}{'설정':<32}{'검증 오차':>10}{'epoch':>7}")
    for i, r in enumerate(results, 1):
        print(f"{i:<4}{r['name']:<32}{r['val']:>9.3f}°{r['epoch']:>7}")

    base = next(r for r in results if r["name"].startswith("base"))
    win = results[0]
    print(f"\n고른 설정: {win['name']}  "
          f"(검증 {base['val']:.3f}° → {win['val']:.3f}°, {win['val'] - base['val']:+.3f}°)")

    path = os.path.join(args.out, "search.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
