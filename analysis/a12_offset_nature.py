"""A12 — 사람 offset의 정체는 무엇인가. **[진단용]** (자율 탐색)

Q-01의 가설: offset이 시선축과 광축의 차이(kappa angle) 같은 **사람의 해부학적 상수**라면
  (H1) 같은 사람 안에서 day가 달라도 보존되어야 한다
  (H2) 좌·우눈에서 (정렬 후) 같은 부호로 나타나야 한다
  (H3) 그 사람의 head pose 습관이나 gaze 평균 같은 '행동' 변수로는 설명되지 않아야 한다

반례 가설: offset은 그 사람의 데이터 분포가 학습 분포에서 벗어난 결과일 뿐이다.
그렇다면 (H3)에서 행동 변수와 강하게 묶여야 한다.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, load_run, pearson, plt, put_numbers,
                             save_fig, save_table, section)

FLIP = np.array([-1.0, 1.0], dtype=np.float32)


def run(run_dir="results/final", tag="runB"):
    section(f"A12 — offset의 정체 [진단용] ({run_dir})")
    import dataset
    d = load_run(run_dir)
    df = dataset.label_frame()
    person, pred, gaze = d["person"], d["pred"], d["gaze"]
    resid = DEG(pred - gaze)
    day = df["day"].to_numpy()
    eye = df["which_eye"].to_numpy()
    head = DEG(d["head"])

    off = np.array([resid[person == p].mean(0) for p in PERSONS])

    # H1 — day 간 보존
    print("H1) 같은 사람 안에서 day가 달라도 offset이 보존되나")
    rows_h1 = []
    split_r = {"yaw": [], "pitch": []}
    rng = np.random.default_rng(0)
    for p in PERSONS:
        m = person == p
        days = np.array(sorted(set(day[m])))
        per_day = np.array([resid[m & (day == dd)].mean(0) for dd in days])
        n_per = np.array([int((m & (day == dd)).sum()) for dd in days])
        keep = n_per >= 20
        if keep.sum() < 4:
            continue
        pd_, nd = per_day[keep], n_per[keep]
        # day를 반으로 갈라 두 반쪽의 offset이 일치하는지 (split-half)
        idx = rng.permutation(len(pd_))
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
        ha = np.average(pd_[a], axis=0, weights=nd[a])
        hb = np.average(pd_[b], axis=0, weights=nd[b])
        rows_h1.append([p, int(keep.sum()),
                        round(float(pd_[:, 0].std(ddof=1)), 3),
                        round(float(pd_[:, 1].std(ddof=1)), 3),
                        round(float(ha[0]), 3), round(float(hb[0]), 3),
                        round(float(ha[1]), 3), round(float(hb[1]), 3)])
        split_r["yaw"].append((ha[0], hb[0]))
        split_r["pitch"].append((ha[1], hb[1]))

    for axis in ("yaw", "pitch"):
        arr = np.array(split_r[axis])
        r, pv = pearson(arr[:, 0], arr[:, 1])
        # Spearman-Brown으로 전체 신뢰도 보정
        rel = 2 * r / (1 + r) if r > -1 else np.nan
        print(f"  {axis:<6} 반쪽-반쪽 상관 r = {r:+.3f} (p={pv:.4f}), "
              f"Spearman-Brown 신뢰도 {rel:.3f}")
        put_numbers(**{f"{tag}_a12_splithalf_{axis}_r": round(r, 4),
                       f"{tag}_a12_splithalf_{axis}_reliability": round(float(rel), 4)})
    within_day_sd = np.array([[r[2], r[3]] for r in rows_h1])
    between_person_sd = off.std(0, ddof=1)
    print(f"  day 간 offset 표준편차(사람 안 중앙값): "
          f"yaw {np.median(within_day_sd[:, 0]):.2f}°  pitch {np.median(within_day_sd[:, 1]):.2f}°")
    print(f"  사람 간 offset 표준편차:                "
          f"yaw {between_person_sd[0]:.2f}°  pitch {between_person_sd[1]:.2f}°")
    save_table(f"a12_h1_day_stability_{tag}",
               ["person", "n_days_used", "day_sd_yaw", "day_sd_pitch",
                "half_a_yaw", "half_b_yaw", "half_a_pitch", "half_b_pitch"], rows_h1)

    # H2 — 좌·우눈에서 같은 부호인가 (정렬 후)
    print("\nH2) 좌·우눈에서 같은 부호로 나타나나 (왼눈 기준 정렬)")
    off_l = np.array([resid[(person == p) & (eye == "left")].mean(0) for p in PERSONS])
    off_r = np.array([(DEG(pred[(person == p) & (eye == "right")] * FLIP
                           - gaze[(person == p) & (eye == "right")] * FLIP)).mean(0)
                      for p in PERSONS])
    rows_h2 = []
    for k, axis in enumerate(("yaw", "pitch")):
        r, pv = pearson(off_l[:, k], off_r[:, k])
        same = int((np.sign(off_l[:, k]) == np.sign(off_r[:, k])).sum())
        print(f"  {axis:<6} 좌 ↔ 우 상관 r = {r:+.3f} (p={pv:.4f}), "
              f"부호 일치 {same}/15명")
        rows_h2.append([axis, round(r, 4), round(pv, 5), same])
        put_numbers(**{f"{tag}_a12_lr_offset_{axis}_r": round(r, 4),
                       f"{tag}_a12_lr_offset_{axis}_sign_match": same})
    save_table(f"a12_h2_left_right_{tag}",
               ["axis", "pearson_r", "p", "sign_match_of_15"], rows_h2)

    # H3 — 행동 변수로 설명되나
    print("\nH3) 그 사람의 행동 변수로 설명되나")
    behav = {
        "head pose 평균 yaw": np.array([head[person == p, 0].mean() for p in PERSONS]),
        "head pose 평균 pitch": np.array([head[person == p, 1].mean() for p in PERSONS]),
        "head pose 퍼짐": np.array([np.hypot(*head[person == p].std(0)) for p in PERSONS]),
        "gaze 평균 yaw": np.array([DEG(gaze[person == p, 0]).mean() for p in PERSONS]),
        "gaze 평균 pitch": np.array([DEG(gaze[person == p, 1]).mean() for p in PERSONS]),
        "day 수": np.array([len(set(day[person == p])) for p in PERSONS], float),
    }
    rows_h3 = []
    print(f"  {'행동 변수':<20}{'↔ offset yaw':>16}{'↔ offset pitch':>17}")
    for name, v in behav.items():
        r1, p1 = pearson(v, off[:, 0]); r2, p2 = pearson(v, off[:, 1])
        s1 = "*" if p1 < 0.05 else " "; s2 = "*" if p2 < 0.05 else " "
        print(f"  {name:<20}  r={r1:+.3f}{s1}(p={p1:.3f})  r={r2:+.3f}{s2}(p={p2:.3f})")
        rows_h3.append([name, round(r1, 4), round(p1, 5), round(r2, 4), round(p2, 5)])
    save_table(f"a12_h3_behaviour_{tag}",
               ["variable", "r_vs_offset_yaw", "p_yaw", "r_vs_offset_pitch", "p_pitch"],
               rows_h3)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    arr = np.array(split_r["pitch"])
    axes[0].scatter(arr[:, 0], arr[:, 1], s=30, c="#1f4e79")
    lim = [arr.min() * 1.1, arr.max() * 1.1]
    axes[0].plot(lim, lim, "--", lw=0.8, c="k")
    r, _ = pearson(arr[:, 0], arr[:, 1])
    axes[0].set(xlabel="day 절반 A의 pitch offset (°)",
                ylabel="day 절반 B의 pitch offset (°)")
    axes[0].text(0.03, 0.95, f"r = {r:+.3f}", transform=axes[0].transAxes,
                 fontsize=7, va="top")
    axes[1].scatter(off_l[:, 1], off_r[:, 1], s=30, c="#1f4e79")
    for i, p in enumerate(PERSONS):
        axes[1].annotate(p, (off_l[i, 1], off_r[i, 1]), fontsize=6,
                         xytext=(3, 2), textcoords="offset points")
    lim = [min(off_l[:, 1].min(), off_r[:, 1].min()) * 1.1,
           max(off_l[:, 1].max(), off_r[:, 1].max()) * 1.1]
    axes[1].plot(lim, lim, "--", lw=0.8, c="k")
    axes[1].set(xlabel="왼눈 pitch offset (°)", ylabel="오른눈 pitch offset (°, 정렬)")
    names = [r[0] for r in rows_h3]
    axes[2].barh(np.arange(len(rows_h3)) - 0.2, [abs(r[1]) for r in rows_h3], 0.38,
                 label="offset yaw", color="#1f4e79")
    axes[2].barh(np.arange(len(rows_h3)) + 0.2, [abs(r[3]) for r in rows_h3], 0.38,
                 label="offset pitch", color="#8aa0b8")
    axes[2].set_yticks(np.arange(len(rows_h3)), names, fontsize=6)
    axes[2].set(xlabel="|상관계수|"); axes[2].legend(fontsize=6); axes[2].invert_yaxis()
    save_fig(fig, "diag_25_offset_nature.png",
             {"person": PERSONS, "off_left_pitch": off_l[:, 1],
              "off_right_pitch": off_r[:, 1]})


if __name__ == "__main__":
    run()
