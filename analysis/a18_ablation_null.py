"""A18 — ablation의 Δ를 '같은 설정의 seed 쌍'과 견준다. (자율 탐색)

관찰: ablation 네 항목이 **전부** 기준(P1 seed 0)보다 명목상 좋게 나왔다.
설계 변경이 모두 도움이 된다기보다, 기준으로 쓴 seed 0이 높은 쪽 뽑기였을 가능성이 크다.

경험적 귀무분포: **같은 설정**을 seed만 바꿔 돌린 쌍들의 Δ와 p를 모은다.
ablation의 Δ가 이 분포 안에 들어가면 "설계 효과"로 읽을 수 없다.
"""
from __future__ import annotations

import glob
import itertools
import os

import numpy as np

from analysis.common import (PERSONS, angular_error, load_folds, load_run, paired,
                             plt, put_numbers, save_fig, save_table, section)

ABLATIONS = {
    "ab_nohead": "head pose 제거",
    "ab_pixnorm": "픽셀별 정규화",
    "ab_angular": "angular 손실",
    "ab_l1": "L1 손실",
    "ab_k5": "5×5 커널",
    "ab_pool2": "풀링 2회",
    "ab_dropout02": "dropout 0.2",
    "ab_narrow": "narrow (16,32,64)",
}
BASE_GLOB = ["results/p1_seed*", "results/p4_seed*"]


def fold_err(run_dir):
    if not os.path.exists(os.path.join(run_dir, "lopo.json")):
        return None
    load_run(run_dir)
    f = load_folds(run_dir, keys=["pred", "gaze"])
    if len(f) < len(PERSONS):
        return None
    return np.array([angular_error(f[p]["pred"], f[p]["gaze"]).mean() for p in PERSONS])


def run():
    section("A18 — ablation Δ를 seed 잡음과 견주기")
    seeds = {}
    for g in BASE_GLOB:
        for d in sorted(glob.glob(g)):
            v = fold_err(d)
            if v is not None:
                seeds[os.path.basename(d)] = v
    print(f"기준 설정의 seed 실행 {len(seeds)}개: {list(seeds)}")
    if len(seeds) < 2:
        print("  seed가 2개 미만이라 귀무분포를 만들 수 없다.")
        return

    # 귀무: 같은 설정, 다른 seed
    null_d, null_p = [], []
    names = list(seeds)
    for a, b in itertools.combinations(names, 2):
        r = paired(seeds[a], seeds[b])
        null_d.append(r["delta"]); null_p.append(r["p"])
    null_d = np.array(null_d); null_p = np.array(null_p)
    print(f"\n같은 설정 · 다른 seed 쌍 {len(null_d)}개 (경험적 귀무)")
    print(f"  |Δ| 중앙값 {np.median(np.abs(null_d)):.3f}°, "
          f"최대 {np.abs(null_d).max():.3f}°")
    print(f"  p < 0.05로 '유의'하게 나온 쌍: {int((null_p < 0.05).sum())}/{len(null_p)}  "
          f"← 같은 설정인데도 이만큼 유의가 나온다")

    base_mean = np.mean([seeds[k] for k in names], axis=0)
    print(f"\n기준 seed별 평균: " + ", ".join(f"{k}={seeds[k].mean():.3f}" for k in names))
    print(f"기준 seed 평균의 평균: {base_mean.mean():.3f}°, "
          f"seed 간 표준편차 {np.std([seeds[k].mean() for k in names], ddof=1):.3f}°")

    rows = []
    print(f"\n{'항목':<18}{'Δ(seed0 기준)':>14}{'Δ(seed 평균 기준)':>18}"
          f"{'|Δ| 귀무 분위':>14}{'판정':>12}")
    s0 = seeds.get("p1_seed0")
    for key, label in ABLATIONS.items():
        v = fold_err(f"results/{key}")
        if v is None:
            continue
        d0 = paired(v, s0)["delta"] if s0 is not None else np.nan
        rm = paired(v, base_mean)
        pct = float((np.abs(null_d) < abs(rm["delta"])).mean() * 100)
        verdict = "seed 잡음 범위" if pct < 95 else "잡음보다 큼"
        print(f"{label:<18}{d0:>+14.3f}{rm['delta']:>+18.3f}"
              f"{pct:>13.0f}%{verdict:>12}")
        rows.append([label, round(float(d0), 4), round(float(rm["delta"]), 4),
                     round(float(rm["se"]), 4), round(float(rm["p"]), 5),
                     round(pct, 1), verdict])

    save_table("a18_ablation_vs_seed_null",
               ["item", "delta_vs_seed0", "delta_vs_seed_mean", "se", "p",
                "percentile_in_null", "verdict"], rows)
    put_numbers(
        a18_n_base_seeds=len(seeds),
        a18_null_abs_delta_median=round(float(np.median(np.abs(null_d))), 4),
        a18_null_abs_delta_max=round(float(np.abs(null_d).max()), 4),
        a18_null_signif_pairs=int((null_p < 0.05).sum()),
        a18_null_n_pairs=len(null_p),
        a18_base_seed_sd=round(float(np.std([seeds[k].mean() for k in names], ddof=1)), 4),
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.axvspan(-np.abs(null_d).max(), np.abs(null_d).max(), color="#dce6f1",
               label=f"같은 설정 seed 쌍의 |Δ| 범위 (n={len(null_d)})")
    ax.axvline(0, lw=0.8, c="k")
    for i, r in enumerate(rows):
        ax.scatter(r[2], i, s=45, c="#c0504d" if r[6] == "잡음보다 큼" else "#1f4e79",
                   zorder=3)
        ax.errorbar(r[2], i, xerr=r[3], fmt="none", ecolor="#1f4e79", capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.set(xlabel="기준(seed 평균) 대비 LOPO 오차 변화 Δ (°)")
    ax.invert_yaxis(); ax.legend(fontsize=6)
    save_fig(fig, "31_ablation_vs_null.png",
             {"item": [r[0] for r in rows], "delta": [r[2] for r in rows],
              "se": [r[3] for r in rows]})

    print(f"\n결론: 같은 설정끼리도 |Δ|가 최대 {np.abs(null_d).max():.3f}°까지 나고, "
          f"{int((null_p<0.05).sum())}/{len(null_p)} 쌍이 p<0.05로 '유의'하게 나온다. "
          "단일 seed ablation의 Δ는 이 폭 안에서 읽어야 한다.")


if __name__ == "__main__":
    run()
