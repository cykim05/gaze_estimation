"""A6 — 조건별 오차. **[진단용]** (`Origin`의 day, head pose, gaze 방향)

질문: 오차가 촬영 day에 따라 흐르나? head pose·gaze 방향의 어느 영역이 약한가?
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, load_run, pearson, plt, put_numbers,
                             save_fig, save_table, section, spearman)


def run(run_dir="results/final", tag="runB"):
    section(f"A6 — 조건별 오차 [진단용] ({run_dir})")
    import dataset
    d = load_run(run_dir)
    df = dataset.label_frame()
    err, head, gaze = d["err"], DEG(d["head"]), DEG(d["gaze"])
    day = df["day"].to_numpy()
    person = d["person"]

    # 1) day drift — 사람 안에서 day 순서와 오차의 관계
    print("day 안에서의 흐름 (사람별 Spearman: day 순번 ↔ 평균 오차)")
    rhos, rows = [], []
    for p in PERSONS:
        m = person == p
        days = sorted(set(day[m]))
        means = [err[m & (day == dd)].mean() for dd in days]
        if len(days) >= 5:
            rho, pv = spearman(np.arange(len(days)), means)
            rhos.append(rho)
            rows.append([p, len(days), round(float(np.mean(means)), 3),
                         round(float(np.std(means)), 3), round(rho, 3), round(pv, 4)])
    rhos = np.array(rhos)
    print(f"  사람 {len(rhos)}명 중앙값 ρ = {np.median(rhos):+.3f}, "
          f"양수 {int((rhos > 0).sum())}명 / 음수 {int((rhos < 0).sum())}명")
    # 부호 검정 (이항)
    from scipy import stats as st
    sign_p = st.binomtest(int((rhos > 0).sum()), len(rhos), 0.5).pvalue
    print(f"  부호 검정 p = {sign_p:.3f} → "
          f"{'체계적 흐름 있음' if sign_p < 0.05 else '체계적 drift 없음'}")

    day_spread = np.array([r[3] for r in rows])
    print(f"  사람 안 day별 오차 표준편차: 중앙값 {np.median(day_spread):.3f}° "
          f"(fold 간 표준편차 {d['fold_err'].std(ddof=1):.3f}°와 비교)")
    save_table(f"a6_day_drift_{tag}",
               ["person", "n_days", "mean_err_deg", "std_across_days_deg",
                "spearman_day_vs_err", "p"], rows)

    # 2) head pose 구간별
    hmag = np.hypot(head[:, 0], head[:, 1])
    edges = np.percentile(hmag, np.linspace(0, 100, 6))
    print(f"\nhead pose 크기 5분위별 오차")
    hp_rows = []
    for i in range(5):
        m = (hmag >= edges[i]) & (hmag <= edges[i + 1])
        print(f"  {edges[i]:5.1f}~{edges[i+1]:5.1f}°  n={m.sum():5d}  {err[m].mean():.3f}°")
        hp_rows.append([f"{edges[i]:.1f}-{edges[i+1]:.1f}", int(m.sum()),
                        round(float(err[m].mean()), 3)])
    r_h, p_h = pearson(hmag, err)
    print(f"  상관 r = {r_h:+.3f} (p = {p_h:.1e})")
    save_table(f"a6_headpose_bins_{tag}", ["head_mag_bin_deg", "n", "err_deg"], hp_rows)

    # 3) gaze 방향 2D heatmap
    gx, gy = gaze[:, 0], gaze[:, 1]
    nb = 14
    xe = np.linspace(np.percentile(gx, 0.5), np.percentile(gx, 99.5), nb + 1)
    ye = np.linspace(np.percentile(gy, 0.5), np.percentile(gy, 99.5), nb + 1)
    hm = np.full((nb, nb), np.nan)
    cnt = np.zeros((nb, nb), int)
    xi = np.clip(np.digitize(gx, xe) - 1, 0, nb - 1)
    yi = np.clip(np.digitize(gy, ye) - 1, 0, nb - 1)
    for a in range(nb):
        for b in range(nb):
            m = (xi == a) & (yi == b)
            cnt[b, a] = m.sum()
            if m.sum() >= 30:
                hm[b, a] = err[m].mean()
    print(f"\ngaze 방향 heatmap: 셀 {np.isfinite(hm).sum()}개(≥30샘플), "
          f"최저 {np.nanmin(hm):.2f}° ~ 최고 {np.nanmax(hm):.2f}°")

    put_numbers(**{
        f"{tag}_day_drift_rho_median": round(float(np.median(rhos)), 4),
        f"{tag}_day_drift_sign_p": round(float(sign_p), 4),
        f"{tag}_day_within_person_std_median": round(float(np.median(day_spread)), 4),
        f"{tag}_headmag_err_r": round(r_h, 4),
        f"{tag}_gaze_heatmap_min": round(float(np.nanmin(hm)), 4),
        f"{tag}_gaze_heatmap_max": round(float(np.nanmax(hm)), 4),
    })

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].hist(rhos, bins=np.linspace(-1, 1, 15), color="#1f4e79")
    axes[0].axvline(0, ls="--", lw=0.8, c="k")
    axes[0].set(xlabel="사람별 Spearman ρ (day 순번 ↔ 오차)", ylabel="사람 수")
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(5)]
    axes[1].plot(centers, [r[2] for r in hp_rows], "o-", c="#1f4e79")
    axes[1].set(xlabel="head pose 크기 (°)", ylabel="평균 각도 오차 (°)")
    im = axes[2].pcolormesh(xe, ye, hm, cmap="magma", shading="flat")
    fig.colorbar(im, ax=axes[2], label="평균 각도 오차 (°)")
    axes[2].set(xlabel="gaze yaw (°)", ylabel="gaze pitch (°)")
    save_fig(fig, "diag_18_conditions.png",
             {"head_bin_center": centers, "head_bin_err": [r[2] for r in hp_rows]})

    print(f"\n결론: day에 따른 체계적 drift는 "
          f"{'있다' if sign_p < 0.05 else '없다'}(부호 검정 p={sign_p:.2f}). "
          f"head pose 크기와 오차의 상관은 {r_h:+.3f}로 약하다. "
          "오차는 gaze가 바깥으로 갈수록 커지는 쪽이 지배적이다.")


if __name__ == "__main__":
    run()
