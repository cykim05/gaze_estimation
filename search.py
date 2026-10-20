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

from common.backend import pick_devices
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

# 1·2단계에서 고른 설정의 best epoch가 20 중 7~8이었다. 코사인 감쇠가 끝나기 전에
# 최저점이 나온다는 뜻이고, 그러면 예산을 줄여 감쇠를 다 쓰는 쪽이 나을 수 있다.
# 학습률과 예산은 서로 묶여 있으므로 같이 본다.
SCHEDULE = {
    f"{e} epoch · lr {lr:g}": dict(epochs=e, lr=lr)
    for e in (10, 20, 40) for lr in (1e-3, 2e-3, 4e-3)
}

# 3단계. 여기까지의 진단은 "용량도 최적화도 아니고 사람 간 일반화가 병목"이다.
# 그 병목을 직접 겨냥하는 기법만 모았다. 전부 교재 계층 규약으로 직접 구현했고
# 수치 미분으로 검증했다(tests/test_layers.py, tests/test_model.py).
# 추론 경로는 그대로 이미지 + head pose뿐이고, 사람 라벨은 학습용 12명에게만 쓴다.
#
# mixup은 사람 라벨을 섞어 버리므로 adv·dro와 같이 쓸 수 없다. 조합에서 뺐다.
GENERALIZATION = {
    "없음 (기준)":           dict(),
    "GroupNorm":            dict(norm="group"),
    "mixup 0.2":            dict(mixup=0.2),
    "mixup 0.4":            dict(mixup=0.4),
    "EMA 0.99":             dict(ema=0.99),
    "EMA 0.999":            dict(ema=0.999),
    "adv 0.3":              dict(adv=0.3),
    "adv 1.0":              dict(adv=1.0),
    "GroupDRO 0.01":        dict(dro=0.01),
    "GN + EMA 0.99":        dict(norm="group", ema=0.99),
    "GN + mixup 0.2":       dict(norm="group", mixup=0.2),
    "GN + adv 0.3":         dict(norm="group", adv=0.3),
    "EMA 0.99 + mixup 0.2": dict(ema=0.99, mixup=0.2),
    "GN + EMA + mixup 0.2": dict(norm="group", ema=0.99, mixup=0.2),
    "GN + EMA + adv 0.3":   dict(norm="group", ema=0.99, adv=0.3),
    "GN + EMA + DRO":       dict(norm="group", ema=0.99, dro=0.01),
}

# 측정 오차를 모르면 순위표를 읽을 수 없다. 같은 설정을 seed만 바꿔 돌려
# 검증 오차가 실행마다 얼마나 흔들리는지 잰다. 이 폭보다 작은 차이는 신호가 아니다.
# 검증이 2명(6,000개)뿐이라 흔들릴 여지가 크다.
SEEDS = {f"seed {s}": dict(seed=s) for s in range(6)}

# 4단계. 1~3단계는 전부 seed 하나로 줄을 세웠는데, seeds 단계에서 같은 설정의
# 실행 간 표준편차가 0.13°로 나왔다. 그 폭이 우리가 비교하던 차이와 같은 크기다.
# 그래서 정말 알고 싶은 것만 남기고 seed를 여러 개 돌려 평균으로 비교한다.
#   (a) 정규화가 실제로 듣는가           plain vs reg
#   (b) 일반화 기법 중 듣는 것이 있는가   reg vs reg+adv / reg+mixup
FINAL = {
    "plain (1차 설정)":    dict(dropout=0.2, weight_decay=0.0, augment=False),
    "dropout 0.5":        dict(dropout=0.5, weight_decay=0.0, augment=False),
    "reg":                dict(dropout=0.35, weight_decay=3e-4, augment=True),
    "reg + adv 1.0":      dict(dropout=0.35, weight_decay=3e-4, augment=True, adv=1.0),
    "reg + mixup 0.2":    dict(dropout=0.35, weight_decay=3e-4, augment=True, mixup=0.2),
}

# B1 (PREREG 2절). lr = 2e-3*sqrt(B/128), B>=512는 첫 epoch linear warmup.
# epoch 예산 20 고정. 테스트는 계산하지 않는다.
BATCH = {
    f"batch {b}": dict(batch_size=b, lr=2e-3 * (b / 128) ** 0.5,
                       warmup_epochs=1.0 if b >= 512 else 0.0, dropout=0.5)
    for b in (128, 256, 512, 1024)
}

PRESETS = {"regularization": REGULARIZATION, "capacity": CAPACITY,
           "schedule": SCHEDULE, "generalization": GENERALIZATION,
           "seeds": SEEDS, "final": FINAL, "batch": BATCH}

SEED_SEP = " · seed "


def _worker(args):
    name, cfg, person, log_dir, device_id = args
    if device_id is not None:
        from common.backend import xp
        xp.cuda.Device(device_id).use()   # GPU 여러 장에 후보를 나눠 건다
    r = train_fold(cfg, person, log_dir=log_dir, track_test=False, verbose=False)
    best = min(r["history"], key=lambda h: h["val"])
    print(f"  {name:<24} 검증 {best['val']:.3f}°  (epoch {best['epoch']:>2}, "
          f"학습 손실 {best['loss']:.4f}, {r['seconds'] / 60:.1f}분"
          + (f", gpu{device_id})" if device_id is not None else ")"), flush=True)
    return {"name": name, "val": best["val"], "epoch": best["epoch"],
            "history": r["history"], "config": {k: v for k, v in cfg.__dict__.items()}}


def _report_with_seeds(results, n_seeds: int) -> None:
    """같은 설정의 여러 seed를 묶어 평균±표준편차로 보고한다.

    설정 사이의 차이를 **실행 간 흔들림과 같은 자에 놓고** 본다. 평균 차이가
    흔들림보다 작으면 순위는 의미가 없다.
    """
    import numpy as np

    groups = {}
    for r in results:
        name = r["name"].split(SEED_SEP)[0]
        groups.setdefault(name, []).append(r["val"])

    rows = []
    for name, vals in groups.items():
        v = np.array(vals)
        rows.append((name, v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v), v))
    rows.sort(key=lambda t: t[1])

    print(f"\n{'순위':<4}{'설정':<20}{'평균':>9}{'표준편차':>10}{'n':>4}   개별 값")
    for i, (name, mu, sd, k, v) in enumerate(rows, 1):
        print(f"{i:<4}{name:<20}{mu:>8.3f}°{sd:>9.3f}°{k:>4}   "
              + " ".join(f"{x:.2f}" for x in sorted(v)))

    best_name, best_mu, best_sd, best_k, best_v = rows[0]
    pooled = np.sqrt(np.mean([sd ** 2 for _, _, sd, k, _ in rows if k > 1]))
    print(f"\n실행 간 표준편차(합산) {pooled:.3f}°")
    print(f"1위 {best_name} {best_mu:.3f}°")
    for name, mu, sd, k, v in rows[1:]:
        gap = mu - best_mu
        se = np.sqrt(best_sd ** 2 / best_k + sd ** 2 / max(k, 1)) or 1e-9
        mark = "유의" if abs(gap) > 2 * se else "구분 불가"
        print(f"  vs {name:<20} {gap:+.3f}°  (평균 차 / 표준오차 {abs(gap) / se:4.1f}) → {mark}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="p00", help="개발용 fold. 이 fold의 테스트는 보지 않는다")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--preset", choices=sorted(PRESETS), default="regularization")
    p.add_argument("--reg", nargs="*", default=[],
                   help="capacity 단계에서 고정할 정규화 설정, 예: dropout=0.5 weight_decay=1e-4 augment=1")
    p.add_argument("--jobs", type=int, default=0, help="0이면 쓸 수 있는 GPU 자리 수만큼")
    p.add_argument("--per-gpu", type=int, default=2, help="GPU 한 장에 동시에 올릴 후보 수")
    p.add_argument("--min-free", type=float, default=6.0, help="쓸 GPU의 최소 여유 메모리 (GiB)")
    p.add_argument("--only", nargs="*", default=[], help="이름에 이 문자열이 들어간 후보만")
    p.add_argument("--seeds", type=int, default=1,
                   help="후보마다 돌릴 seed 수. 2 이상이면 평균±표준편차로 비교한다")
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

    if args.only:
        candidates = {k: v for k, v in candidates.items()
                      if any(o in k for o in args.only)}
        print(f"후보를 {len(candidates)}개로 좁혔다: {list(candidates)}")

    if args.seeds > 1:
        candidates = {f"{name}{SEED_SEP}{s}": {**kwargs, "seed": s}
                      for name, kwargs in candidates.items()
                      for s in range(args.seeds)}

    args.out = args.out or f"results/search-{args.preset}"
    os.makedirs(args.out, exist_ok=True)

    def slug(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z.]+", "-", name).strip("-")

    from common.backend import describe
    devices = pick_devices(len(candidates), args.per_gpu, args.min_free)

    # 후보가 epochs를 직접 정하면(schedule 단계) 그쪽을 쓴다
    jobs = [(name, Config(**{"epochs": args.epochs, **kwargs}), args.fold,
             os.path.join(args.out, slug(name)), dev)
            for (name, kwargs), dev in zip(candidates.items(), devices)]

    workers = args.jobs or min(len(jobs), len({d for d in devices if d is not None})
                               * args.per_gpu or len(jobs))
    budgets = sorted({j[1].epochs for j in jobs})
    print(f"개발용 fold {args.fold} · 후보 {len(jobs)}개 · "
          f"{'/'.join(map(str, budgets))} epoch · worker {workers}개 · "
          f"쓰는 GPU {sorted({d for d in devices if d is not None})}")
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

    if args.seeds > 1:
        _report_with_seeds(results, args.seeds)
    else:
        print(f"\n{'순위':<4}{'설정':<26}{'검증 오차':>10}{'epoch':>7}")
        for i, r in enumerate(results, 1):
            print(f"{i:<4}{r['name']:<26}{r['val']:>9.3f}°{r['epoch']:>7}")
        base = next((r for r in results if r["name"].strip().startswith("base")), results[-1])
        win = results[0]
        print(f"\n고른 설정: {win['name']}  "
              f"(기준 {base['name']} {base['val']:.3f}° → {win['val']:.3f}°, "
              f"{win['val'] - base['val']:+.3f}°)")
        print("\n주의: seed 하나로 잰 값이다. 같은 설정의 실행 간 표준편차가 0.13° 수준이라")
        print("      이보다 작은 차이는 읽으면 안 된다. --seeds 로 여러 번 돌려 평균을 본다.")

    path = os.path.join(args.out, "search.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
