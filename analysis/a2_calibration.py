"""A2 — few-shot calibration 곡선. **[진단용]**

질문: 테스트 사람의 라벨 k장으로 offset(2개 파라미터)만 추정해 빼면 몇 장에서 포화하나?
테스트 라벨을 쓰므로 메인 LOPO 수치와 분리한다. 예측 경로에는 절대 쓰지 않는다.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, angular_error, load_run, plt,
                             put_numbers, save_fig, save_table, section)

KS = [0, 1, 3, 10, 30, 100]
BOOT = 200


def run(run_dir="results/final", tag="runB"):
    section(f"A2 — few-shot calibration [진단용] ({run_dir})")
    d = load_run(run_dir)
    person, pred, gaze = d["person"], d["pred"], d["gaze"]
    rng = np.random.default_rng(0)

    curves = {k: [] for k in KS}          # k → bootstrap 반복별 15-fold 평균
    for b in range(BOOT):
        for k in KS:
            fold_means = []
            for p in PERSONS:
                m = np.flatnonzero(person == p)
                if k == 0:
                    shift = np.zeros(2)
                    evalu = m
                else:
                    pick = rng.choice(m, size=min(k, len(m)), replace=False)
                    shift = (pred[pick] - gaze[pick]).mean(0)
                    # 보정에 쓴 샘플은 평가에서 뺀다 (낙관 방지)
                    evalu = np.setdiff1d(m, pick, assume_unique=False)
                fold_means.append(angular_error(pred[evalu] - shift, gaze[evalu]).mean())
            curves[k].append(float(np.mean(fold_means)))

    print(f"{'k':>5}{'평균 (°)':>11}{'95% CI':>20}{'k=0 대비':>12}")
    rows, stats_out = [], {}
    base = np.mean(curves[0])
    for k in KS:
        v = np.array(curves[k])
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"{k:>5}{v.mean():>11.3f}   [{lo:.3f}, {hi:.3f}]{v.mean() - base:>+12.3f}")
        rows.append([k, round(float(v.mean()), 4), round(float(lo), 4),
                     round(float(hi), 4), round(float(v.mean() - base), 4)])
        stats_out[k] = (v.mean(), lo, hi)

    # 오라클(전부 사용) 대비 몇 %를 회수하나
    full = np.array([(pred[person == p] - gaze[person == p]).mean(0) for p in PERSONS])
    oracle_err = np.mean([
        angular_error(pred[person == p] - full[i], gaze[person == p]).mean()
        for i, p in enumerate(PERSONS)])
    print(f"\n오라클(그 사람 3,000개 전부로 offset 추정) {oracle_err:.3f}°")
    for k in KS[1:]:
        rec = (base - stats_out[k][0]) / (base - oracle_err)
        print(f"  k={k:>3}: 회수율 {rec:.1%}")
        put_numbers(**{f"{tag}_calib_recovery_k{k}": round(float(rec), 4)})

    save_table(f"a2_calibration_{tag}",
               ["k_labels", "mean_deg", "ci95_lo", "ci95_hi", "delta_vs_k0"], rows)
    put_numbers(**{
        f"{tag}_calib_k0": round(float(stats_out[0][0]), 4),
        f"{tag}_calib_k10": round(float(stats_out[10][0]), 4),
        f"{tag}_calib_k100": round(float(stats_out[100][0]), 4),
        f"{tag}_calib_oracle": round(float(oracle_err), 4),
        f"{tag}_calib_bootstrap_n": BOOT,
    })

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = np.array([max(k, 0.5) for k in KS])
    mu = np.array([stats_out[k][0] for k in KS])
    lo = np.array([stats_out[k][1] for k in KS])
    hi = np.array([stats_out[k][2] for k in KS])
    ax.fill_between(xs, lo, hi, alpha=0.25, color="#1f4e79")
    ax.plot(xs, mu, "o-", ms=4, color="#1f4e79", label="offset만 보정")
    ax.axhline(oracle_err, ls="--", lw=0.9, c="#c0504d",
               label=f"오라클 {oracle_err:.2f}°")
    ax.axhline(base, ls=":", lw=0.9, c="k", label=f"보정 없음 {base:.2f}°")
    ax.set(xscale="log", xlabel="보정에 쓴 테스트 사람 라벨 수 k (장, 로그 눈금)",
           ylabel="LOPO 평균 각도 오차 (°)")
    ax.set_xticks(xs); ax.set_xticklabels([str(k) for k in KS])
    ax.legend(fontsize=6)
    save_fig(fig, "diag_13_calibration.png",
             {"k": KS, "mean": mu, "ci_lo": lo, "ci_hi": hi})

    print(f"\n결론: 라벨 {10}장이면 offset 회수의 "
          f"{(base - stats_out[10][0]) / (base - oracle_err):.0%}를 얻는다.")


if __name__ == "__main__":
    run()
