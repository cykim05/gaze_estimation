"""E2 — ensemble 표. (a) 단일 seed (b) epoch-ensemble (c) seed-ensemble (d) 둘 다.

PREREG 6절: epoch-ensemble의 K는 **개발용 fold의 검증으로만** 고른다.
모든 비교는 15-fold 대응표본 t와 p값으로 보고한다.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from analysis.common import (PERSONS, angular_error, load_folds, load_run, paired,
                             plt, put_numbers, save_fig, save_table, section)

DEV_FOLD = "p00"
K_CANDIDATES = [1, 2, 3, 5, 8]


def epoch_ensemble_pred(fold_npz: dict, k: int, which: str) -> np.ndarray:
    """마지막 k epoch의 예측 평균. which: 'test' 또는 'val'."""
    tr = fold_npz[f"trace_{which}_pred"]
    return tr[-k:].mean(0)


def pick_k_on_dev(run_dirs: list[str]) -> tuple[int, list]:
    """개발용 fold의 **검증** 오차로만 K를 고른다. 테스트는 보지 않는다."""
    rows = []
    for k in K_CANDIDATES:
        errs = []
        for rd in run_dirs:
            f = os.path.join(rd, "folds", f"{DEV_FOLD}.npz")
            if not os.path.exists(f):
                continue
            with np.load(f) as z:
                if "trace_val_pred" not in z.files or z["trace_val_pred"].size == 0:
                    continue
                pred = z["trace_val_pred"][-k:].mean(0)
                errs.append(float(angular_error(pred, z["val_gaze"]).mean()))
        if errs:
            rows.append([k, round(float(np.mean(errs)), 4), len(errs)])
    if not rows:
        return 1, rows
    best = min(rows, key=lambda r: r[1])[0]
    return best, rows


def collect(run_dirs: list[str], k: int) -> dict:
    """실행별 fold 예측(단일/epoch-ensemble)을 모은다."""
    out = {}
    for rd in run_dirs:
        folds = load_folds(rd, keys=["pred", "gaze", "trace_test_pred"])
        if len(folds) < len(PERSONS):
            continue
        single, ens = {}, {}
        for p, z in folds.items():
            single[p] = z["pred"]
            ens[p] = (z["trace_test_pred"][-k:].mean(0)
                      if z.get("trace_test_pred") is not None and z["trace_test_pred"].size
                      else z["pred"])
        out[rd] = {"single": single, "epoch_ens": ens,
                   "gaze": {p: z["gaze"] for p, z in folds.items()}}
    return out


def fold_err(pred: dict, gaze: dict) -> np.ndarray:
    return np.array([angular_error(pred[p], gaze[p]).mean() for p in PERSONS])


def run(pattern="results/p1_seed*"):
    section("E2 — ensemble")
    run_dirs = sorted(d for d in glob.glob(pattern) if os.path.exists(f"{d}/lopo.json"))
    print(f"실행 {len(run_dirs)}개: {[os.path.basename(d) for d in run_dirs]}")
    if not run_dirs:
        print("  P1 결과가 아직 없다.")
        return

    for d in run_dirs:
        load_run(d)          # 정합성 검사

    k, k_rows = pick_k_on_dev(run_dirs)
    print(f"\nepoch-ensemble K 선택 (개발용 fold {DEV_FOLD}의 **검증** 오차로만)")
    for kk, v, n in k_rows:
        print(f"  K={kk:<3} 검증 {v:.4f}°  (실행 {n}개 평균){'   ← 선택' if kk == k else ''}")
    put_numbers(e2_epoch_ensemble_K=k,
                e2_K_selection_val=[[r[0], r[1]] for r in k_rows])

    data = collect(run_dirs, k)
    gaze = data[run_dirs[0]]["gaze"]

    single_errs = np.array([fold_err(data[d]["single"], gaze) for d in run_dirs])
    ens_errs = np.array([fold_err(data[d]["epoch_ens"], gaze) for d in run_dirs])

    # seed-ensemble: 예측을 평균
    seed_pred = {p: np.mean([data[d]["single"][p] for d in run_dirs], axis=0)
                 for p in PERSONS}
    both_pred = {p: np.mean([data[d]["epoch_ens"][p] for d in run_dirs], axis=0)
                 for p in PERSONS}
    seed_err = fold_err(seed_pred, gaze)
    both_err = fold_err(both_pred, gaze)

    base = single_errs.mean(0)          # seed 평균 (단일 모델의 기대 성능)
    variants = {
        "(a) 단일 모델 (seed 평균)": base,
        f"(b) epoch-ensemble K={k}": ens_errs.mean(0),
        f"(c) seed-ensemble ({len(run_dirs)})": seed_err,
        "(d) (b)+(c)": both_err,
    }

    print(f"\n{'방식':<28}{'평균':>8}{'Δ':>9}{'SE':>8}{'t':>8}{'p':>9}")
    rows, table_rows = [], []
    for name, v in variants.items():
        r = paired(v, base)
        mark = "" if name.startswith("(a)") else ("유의" if r["significant"] else "구분 불가")
        print(f"{name:<28}{v.mean():>8.3f}{r['delta']:>+9.3f}{r['se']:>8.3f}"
              f"{r['t']:>8.2f}{r['p']:>9.4f}  {mark}")
        rows.append((name, v, r))
        table_rows.append([name] + [round(float(x), 3) for x in v]
                          + [round(float(v.mean()), 4), round(float(r["delta"]), 4),
                             round(float(r["se"]), 4), round(float(r["t"]), 3),
                             round(float(r["p"]), 5)])

    print(f"\n단일 모델 seed 간 편차: 평균 {single_errs.mean(1).mean():.3f}°, "
          f"실행별 {np.round(single_errs.mean(1), 3)}")

    save_table("e2_ensemble",
               ["method"] + list(PERSONS) + ["mean_deg", "delta_vs_single", "se", "t", "p"],
               table_rows)
    put_numbers(
        e2_single_mean=round(float(base.mean()), 4),
        e2_epoch_ens_mean=round(float(ens_errs.mean(0).mean()), 4),
        e2_seed_ens_mean=round(float(seed_err.mean()), 4),
        e2_both_mean=round(float(both_err.mean()), 4),
        e2_n_seeds=len(run_dirs),
        e2_seed_run_means=[round(float(v), 4) for v in single_errs.mean(1)],
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(PERSONS)); w = 0.2
    for i, (name, v, _) in enumerate(rows):
        ax.bar(x + (i - 1.5) * w, v, w, label=name)
    ax.set_xticks(x, PERSONS, rotation=90)
    ax.set(ylabel="fold 평균 각도 오차 (°)")
    ax.legend(fontsize=6)
    save_fig(fig, "20_ensemble.png",
             {"person": PERSONS, **{n: v for n, v, _ in rows}})
    return rows


if __name__ == "__main__":
    run()
