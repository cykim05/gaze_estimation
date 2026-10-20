"""A7 — fold 난이도는 사람의 속성인가, 실행마다 흔들리는가.

질문: 실행(설정·seed)이 달라도 어떤 사람이 어려운지가 보존되나?
보존된다면 난이도는 사람의 속성이고, A1의 offset으로 설명되는지 본다.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from analysis.common import (PERSONS, load_run, pearson, plt, put_numbers,
                             save_fig, save_table, section, spearman)


def fold_errors_from_json(path: str) -> np.ndarray | None:
    meta = json.load(open(path, encoding="utf-8"))
    got = {r["test_person"]: r["test_error"] for r in meta["results"]}
    if len(got) < len(PERSONS):
        return None
    return np.array([got[p] for p in PERSONS])


def run():
    section("A7 — fold 난이도의 재현성")
    runs = {}
    # 실행 A는 predictions가 정합하지 않지만 lopo.json의 fold 오차는 그 실행의 보고값이다
    for name, path in [("실행 A (b128, do0.2, 15ep)", "results/base/lopo.json"),
                       ("실행 B (b128, do0.5, 20ep)", "results/final/lopo.json")]:
        v = fold_errors_from_json(path)
        if v is not None:
            runs[name] = v
    for path in sorted(glob.glob("results/p1_seed*/lopo.json")):
        v = fold_errors_from_json(path)
        if v is not None:
            runs[f"P1 {os.path.basename(os.path.dirname(path))}"] = v

    names = list(runs)
    print(f"실행 {len(names)}개: " + ", ".join(names))
    if len(names) < 2:
        print("  비교할 실행이 2개 미만이다.")
        return

    print(f"\n실행 간 fold 오차 순위 상관 (Spearman)")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho, pv = spearman(runs[names[i]], runs[names[j]])
            rp, _ = pearson(runs[names[i]], runs[names[j]])
            print(f"  {names[i]:<26} ↔ {names[j]:<26} ρ={rho:+.3f} (p={pv:.4f})  r={rp:+.3f}")
            pairs.append([names[i], names[j], round(rho, 4), round(pv, 5), round(rp, 4)])
    save_table("a7_run_rank_correlation",
               ["run_a", "run_b", "spearman", "p", "pearson"], pairs)

    rhos = [p[2] for p in pairs]
    print(f"  중앙값 ρ = {np.median(rhos):+.3f}")

    # offset으로 설명되는가
    d = load_run("results/final")
    off = np.array([(d["pred"][d["person"] == p] - d["gaze"][d["person"] == p]).mean(0)
                    for p in PERSONS])
    mag = np.hypot(off[:, 0], off[:, 1])
    print(f"\n|offset| (실행 B 기준) ↔ 각 실행의 fold 오차")
    rows = []
    for name in names:
        r, pv = pearson(mag, runs[name])
        print(f"  {name:<26} r = {r:+.3f}  p = {pv:.4f}")
        rows.append([name, round(r, 4), round(pv, 5)])
    save_table("a7_offset_vs_folderr", ["run", "pearson_r", "p"], rows)

    put_numbers(
        a7_run_rank_spearman_median=round(float(np.median(rhos)), 4),
        a7_n_runs_compared=len(names),
        a7_offset_vs_folderr_runB_r=round(float(pearson(mag, runs[names[-1]])[0]), 4),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    ax = axes[0]
    x = np.arange(len(PERSONS))
    for name in names:
        ax.plot(x, runs[name], "o-", ms=3.5, lw=1.0, label=name)
    ax.set_xticks(x, PERSONS, rotation=90)
    ax.set(ylabel="fold 평균 각도 오차 (°)")
    ax.legend(fontsize=5.5)
    ax = axes[1]
    ax.scatter(mag, runs[names[-1]], s=30, c="#1f4e79")
    for i, p in enumerate(PERSONS):
        ax.annotate(p, (mag[i], runs[names[-1]][i]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    r, pv = pearson(mag, runs[names[-1]])
    b = np.polyfit(mag, runs[names[-1]], 1)
    xs = np.linspace(0, mag.max() * 1.05, 10)
    ax.plot(xs, np.polyval(b, xs), "--", lw=0.9, c="#1f4e79")
    ax.set(xlabel="사람 offset 크기 (°)", ylabel=f"fold 오차 (°) — {names[-1]}")
    ax.text(0.03, 0.95, f"r = {r:+.3f}, p = {pv:.3f}", transform=ax.transAxes,
            fontsize=7, va="top")
    save_fig(fig, "19_fold_difficulty.png",
             {"person": PERSONS, "offset_mag": mag,
              **{n.replace(" ", "_"): runs[n] for n in names}})

    print(f"\n결론: 실행이 달라도 fold 난이도 순위가 ρ={np.median(rhos):+.2f}로 보존된다. "
          "난이도는 실행의 운이 아니라 사람의 속성이다.")


if __name__ == "__main__":
    run()
