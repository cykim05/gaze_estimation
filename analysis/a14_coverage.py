"""A14 — gaze가 클수록 오차가 커지는 것은 학습 데이터가 없어서인가. **[진단용]**

A3에서 offset을 빼고도 |gaze|가 클수록 오차가 커졌다(r=+0.24).
두 가지 설명이 가능하다.
  (C1) 학습 데이터가 바깥쪽에 드물다 — 데이터 문제. 더 모으면 줄어든다.
  (C2) 바깥쪽 시선은 눈 이미지에 덜 드러난다 — 과제의 성질. 데이터로 안 줄어든다.

구분: 학습 분포의 밀도를 격자로 재고, 같은 밀도 안에서도 |gaze|가 오차를 끌어올리는지 본다.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, load_run, pearson, plt, put_numbers,
                             save_fig, save_table, section)


def run(run_dir="results/final", tag="runB"):
    section(f"A14 — 바깥쪽 시선의 오차: 밀도인가 성질인가 [진단용] ({run_dir})")
    import dataset
    from train import split_persons

    d = load_run(run_dir)
    person, pred, gaze, err = d["person"], d["pred"], d["gaze"], d["err"]
    samples = dataset.load(verbose=False)

    # offset을 뺀 오차로 본다 (A1의 사람 성분을 제거)
    off = np.array([(pred[person == p] - gaze[person == p]).mean(0) for p in PERSONS])
    err_c = np.linalg.norm(np.zeros((len(pred), 1)), axis=1)  # placeholder
    from metrics import angular_error
    err_c = angular_error(pred - off[[PERSONS.index(p) for p in person]], gaze)

    g = DEG(gaze)
    size = np.hypot(g[:, 0], g[:, 1])

    # fold마다 '그 fold의 학습 14명' 분포에서의 밀도를 잰다
    nb = 24
    xe = np.linspace(g[:, 0].min(), g[:, 0].max(), nb + 1)
    ye = np.linspace(g[:, 1].min(), g[:, 1].max(), nb + 1)
    dens = np.zeros(len(gaze))
    for p in PERSONS:
        tr_p, _ = split_persons(p, 2, list(PERSONS))
        tg = DEG(samples.only(*tr_p).gaze)
        H, _, _ = np.histogram2d(tg[:, 0], tg[:, 1], bins=[xe, ye])
        H = H / H.sum()
        m = person == p
        xi = np.clip(np.digitize(g[m, 0], xe) - 1, 0, nb - 1)
        yi = np.clip(np.digitize(g[m, 1], ye) - 1, 0, nb - 1)
        dens[m] = H[xi, yi]

    logd = np.log10(dens + 1e-7)
    r_d, p_d = pearson(logd, err_c)
    r_s, p_s = pearson(size, err_c)
    print(f"offset 제거 오차와의 상관")
    print(f"  log10(학습 밀도)  r = {r_d:+.3f}  (p = {p_d:.1e})")
    print(f"  |gaze|            r = {r_s:+.3f}  (p = {p_s:.1e})")
    print(f"  밀도 ↔ |gaze|     r = {pearson(logd, size)[0]:+.3f}")

    # 밀도를 묶어 놓고 |gaze| 효과가 남는지
    print(f"\n학습 밀도 5분위 안에서 |gaze| 상·하위 절반의 오차")
    edges = np.percentile(logd, [0, 20, 40, 60, 80, 100])
    rows = []
    for i in range(5):
        m = (logd >= edges[i]) & (logd <= edges[i + 1])
        if m.sum() < 100:
            continue
        med = np.median(size[m])
        lo = m & (size <= med); hi = m & (size > med)
        rows.append([f"{edges[i]:.2f}~{edges[i+1]:.2f}", int(m.sum()),
                     round(float(err_c[lo].mean()), 3), round(float(err_c[hi].mean()), 3),
                     round(float(err_c[hi].mean() - err_c[lo].mean()), 3),
                     round(float(np.median(size[lo])), 2), round(float(np.median(size[hi])), 2)])
        print(f"  밀도 {edges[i]:+.2f}~{edges[i+1]:+.2f}  n={m.sum():5d}  "
              f"|gaze| 작은 쪽 {err_c[lo].mean():.2f}°  큰 쪽 {err_c[hi].mean():.2f}°  "
              f"차이 {err_c[hi].mean() - err_c[lo].mean():+.2f}°")
    deltas = np.array([r[4] for r in rows])
    print(f"  → 밀도를 묶어 놓아도 |gaze| 효과가 평균 {deltas.mean():+.2f}° 남는다 "
          f"({int((deltas > 0).sum())}/{len(deltas)} 분위에서 양수)")

    save_table(f"a14_density_{tag}",
               ["log_density_bin", "n", "err_small_gaze", "err_large_gaze",
                "delta", "median_gaze_small", "median_gaze_large"], rows)
    put_numbers(**{
        f"{tag}_a14_density_err_r": round(r_d, 4),
        f"{tag}_a14_gazesize_err_r": round(r_s, 4),
        f"{tag}_a14_gaze_effect_within_density": round(float(deltas.mean()), 4),
        f"{tag}_a14_bins_positive": int((deltas > 0).sum()),
    })

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    q = np.percentile(logd, np.linspace(0, 100, 9))
    cs, es = [], []
    for i in range(8):
        m = (logd >= q[i]) & (logd <= q[i + 1])
        cs.append((q[i] + q[i + 1]) / 2); es.append(err_c[m].mean())
    axes[0].plot(cs, es, "o-", c="#1f4e79")
    axes[0].set(xlabel="log10(해당 fold 학습 분포의 gaze 밀도)",
                ylabel="offset 제거 후 각도 오차 (°)")
    qq = np.percentile(size, np.linspace(0, 100, 9))
    cs2, es2 = [], []
    for i in range(8):
        m = (size >= qq[i]) & (size <= qq[i + 1])
        cs2.append((qq[i] + qq[i + 1]) / 2); es2.append(err_c[m].mean())
    axes[1].plot(cs2, es2, "o-", c="#c0504d")
    axes[1].set(xlabel="|gaze| (°)", ylabel="offset 제거 후 각도 오차 (°)")
    x = np.arange(len(rows)); w = 0.38
    axes[2].bar(x - w / 2, [r[2] for r in rows], w, label="|gaze| 작은 절반", color="#8aa0b8")
    axes[2].bar(x + w / 2, [r[3] for r in rows], w, label="|gaze| 큰 절반", color="#1f4e79")
    axes[2].set_xticks(x, [r[0] for r in rows], rotation=45, fontsize=6)
    axes[2].set(xlabel="학습 밀도 분위", ylabel="각도 오차 (°)")
    axes[2].legend(fontsize=6)
    save_fig(fig, "diag_27_coverage.png",
             {"log_density": cs, "err": es, "gaze_size": cs2, "err_by_size": es2})

    print(f"\n결론: 학습 밀도도 오차와 묶여 있지만(r={r_d:+.2f}), 밀도를 고정해도 "
          f"|gaze| 효과가 {deltas.mean():+.2f}° 남는다. "
          "데이터를 더 모으는 것만으로는 바깥쪽 시선의 오차가 사라지지 않는다 (C1 + C2 둘 다).")


if __name__ == "__main__":
    run()
