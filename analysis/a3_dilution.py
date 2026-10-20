"""A3 — '수축'은 진짜인가, regression dilution인가.

SOLUTION 7.2는 pred~gt 기울기가 1보다 작은 것을 "모델이 평균 쪽으로 몸을 사린다"로
읽었다. 그런데 예측에 잡음이 있으면 **설명변수에 잡음이 있는 회귀**가 되어
기울기가 구조적으로 1보다 작아진다(regression dilution). 방향을 바꿔 보면 갈린다.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, angular_error, load_run, pearson, plt,
                             put_numbers, save_fig, save_table, section)


def run(run_dir="results/final", tag="runB"):
    section(f"A3 — regression dilution ({run_dir})")
    d = load_run(run_dir)
    person, pred, gaze, err = d["person"], d["pred"], d["gaze"], d["err"]
    g, p_ = DEG(gaze), DEG(pred)

    rows = []
    print(f"{'축':<7}{'pred~gt':>10}{'gt~pred':>10}{'fold중앙값':>12}"
          f"{'fold≥1':>9}{'감쇠비':>9}")
    for k, name in enumerate(("yaw", "pitch")):
        fwd = np.polyfit(g[:, k], p_[:, k], 1)[0]
        rev = np.polyfit(p_[:, k], g[:, k], 1)[0]
        per_fold = np.array([np.polyfit(p_[person == q, k], g[person == q, k], 1)[0]
                             for q in PERSONS])
        # 잡음 대 신호로 예측되는 감쇠: slope_fwd ≈ var(signal)/(var(signal)+var(noise))
        var_sig = g[:, k].var()
        var_noise = (p_[:, k] - g[:, k]).var()
        attenuation = var_sig / (var_sig + var_noise)
        n_ge1 = int((per_fold >= 1.0).sum())
        print(f"{name:<7}{fwd:>10.3f}{rev:>10.3f}{np.median(per_fold):>12.3f}"
              f"{n_ge1:>7}/15{attenuation:>9.3f}")
        rows.append([name, round(float(fwd), 4), round(float(rev), 4),
                     round(float(np.median(per_fold)), 4), n_ge1,
                     round(float(attenuation), 4)])
        put_numbers(**{
            f"{tag}_slope_pred_on_gt_{name}": round(float(fwd), 4),
            f"{tag}_slope_gt_on_pred_{name}": round(float(rev), 4),
            f"{tag}_slope_fold_median_{name}": round(float(np.median(per_fold)), 4),
            f"{tag}_folds_slope_ge1_{name}": n_ge1,
            f"{tag}_dilution_predicted_slope_{name}": round(float(attenuation), 4),
        })
    save_table(f"a3_slopes_{tag}",
               ["axis", "slope_pred_on_gt", "slope_gt_on_pred",
                "slope_gt_on_pred_fold_median", "folds_with_slope_ge1",
                "dilution_predicted_slope"], rows)

    # offset을 뺀 뒤에도 gaze 크기에 따라 오차가 커지는가
    off = np.array([(pred[person == q] - gaze[person == q]).mean(0) for q in PERSONS])
    corrected = pred - off[[PERSONS.index(q) for q in person]]
    err_c = angular_error(corrected, gaze)
    size = DEG(np.linalg.norm(gaze, axis=1))
    edges = np.percentile(size, [0, 20, 40, 60, 80, 100])
    print(f"\ngaze 크기 5분위별 평균 오차 (도)")
    print(f"  {'구간':<18}{'원래':>8}{'offset 제거 [진단용]':>22}")
    q_rows = []
    for i in range(5):
        m = (size >= edges[i]) & (size <= edges[i + 1])
        print(f"  {edges[i]:5.1f}~{edges[i+1]:5.1f}°   {err[m].mean():>8.2f}{err_c[m].mean():>22.2f}")
        q_rows.append([f"{edges[i]:.1f}-{edges[i+1]:.1f}", int(m.sum()),
                       round(float(err[m].mean()), 3), round(float(err_c[m].mean()), 3)])
    r0, p0 = pearson(size, err)
    r1, p1 = pearson(size, err_c)
    print(f"  상관: 원래 r={r0:+.3f} (p={p0:.1e})   offset 제거 r={r1:+.3f} (p={p1:.1e})")
    save_table(f"a3_gaze_size_{tag}",
               ["gaze_size_bin_deg", "n", "error_deg", "error_after_offset_deg"], q_rows)
    put_numbers(**{f"{tag}_gazesize_err_r": round(r0, 4),
                   f"{tag}_gazesize_err_r_after_offset": round(r1, 4)})

    # 그림: 두 회귀선
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, k, name in zip(axes[:2], (0, 1), ("yaw", "pitch")):
        ax.hexbin(g[:, k], p_[:, k], gridsize=55, cmap="Greys", bins="log", mincnt=1)
        lim = np.array([g[:, k].min(), g[:, k].max()])
        fwd = np.polyfit(g[:, k], p_[:, k], 1)
        rev = np.polyfit(p_[:, k], g[:, k], 1)
        ax.plot(lim, np.polyval(fwd, lim), "-", lw=1.4, c="#1f4e79",
                label=f"pred~gt  기울기 {fwd[0]:.2f}")
        ys = np.array([p_[:, k].min(), p_[:, k].max()])
        ax.plot(np.polyval(rev, ys), ys, "-", lw=1.4, c="#c0504d",
                label=f"gt~pred  기울기 {rev[0]:.2f}")
        ax.plot(lim, lim, "--", lw=0.8, c="k", label="y = x")
        ax.set(xlabel=f"정답 {name} (°)", ylabel=f"예측 {name} (°)")
        ax.legend(fontsize=6, loc="upper left")
    ax = axes[2]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(5)]
    ax.plot(centers, [r[2] for r in q_rows], "o-", c="#1f4e79", label="원래")
    ax.plot(centers, [r[3] for r in q_rows], "^--", c="#c0504d",
            label="offset 제거 [진단용]")
    ax.set(xlabel="gaze 크기 |gaze| (°)", ylabel="평균 각도 오차 (°)")
    ax.legend(fontsize=6)
    save_fig(fig, "14_regression_dilution.png",
             {"gaze_size_center": centers,
              "err": [r[2] for r in q_rows], "err_after_offset": [r[3] for r in q_rows]})

    print("\n결론: 같은 데이터에서 회귀 방향만 바꾸면 기울기가 1 아래/위로 갈린다. "
          "'수축'은 모델의 성향이라기보다 예측 잡음이 만드는 regression dilution이다.")


if __name__ == "__main__":
    run()
