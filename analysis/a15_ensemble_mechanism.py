"""A15 — ensemble은 어느 성분을 줄이나. (자율 탐색)

A13에서 잔차를 '사람 고정 / 장면 / 눈마다 독립'으로 갈랐다.
seed를 평균하면 **모델 초기화에서 오는 흔들림**만 줄어야 하고, 사람 고정 성분은
seed가 달라도 공유되므로 안 줄어야 한다. 실제로 그런지 본다.

이 분해가 맞다면 ensemble의 한계가 미리 보인다.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from analysis.common import (sig, DEG, PERSONS, angular_error, load_folds, load_run,
                             paired, pearson, plt, put_numbers, save_fig,
                             save_table, section)


def run(pattern=("results/p1_seed*", "results/p4_seed*")):
    section("A15 — ensemble이 줄이는 성분")
    pats = (pattern,) if isinstance(pattern, str) else pattern
    dirs = sorted(d for pat in pats for d in glob.glob(pat)
                  if os.path.exists(f"{d}/lopo.json")
                  and len(glob.glob(f"{d}/folds/*.npz")) >= len(PERSONS))
    print(f"실행 {len(dirs)}개: {[os.path.basename(d) for d in dirs]}")
    if len(dirs) < 2:
        print("  seed가 2개 미만이라 건너뛴다.")
        return

    for d in dirs:
        load_run(d)
    folds = {d: load_folds(d, keys=["pred", "gaze"]) for d in dirs}
    gaze = {p: folds[dirs[0]][p]["gaze"] for p in PERSONS}

    # 사람별 offset이 seed 간에 얼마나 보존되나
    offs = []
    for d in dirs:
        offs.append(np.array([DEG(folds[d][p]["pred"] - gaze[p]).mean(0)
                              for p in PERSONS]))
    offs = np.array(offs)                      # (seed, 15, 2)
    print(f"\n사람 offset이 seed 간에 보존되나 (쌍별 상관)")
    rows = []
    for k, axis in enumerate(("yaw", "pitch")):
        rs = []
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                r, _ = pearson(offs[i, :, k], offs[j, :, k])
                rs.append(r)
        print(f"  {axis:<6} seed 간 offset 상관 중앙값 r = {np.median(rs):+.3f}  "
              f"(쌍 {len(rs)}개)")
        rows.append([axis, round(float(np.median(rs)), 4), len(rs)])
        put_numbers(**{f"a15_offset_seed_corr_{axis}": round(float(np.median(rs)), 4)})
    save_table("a15_offset_seed_corr", ["axis", "median_r_between_seeds", "n_pairs"], rows)

    # 잔차를 '모든 seed가 공유하는 부분'과 '모델마다 다른 부분'으로 가른다
    print(f"\n잔차 분산 분해 (seed {len(dirs)}개)")
    res = np.concatenate([np.concatenate([DEG(folds[d][p]["pred"] - gaze[p])
                                          for p in PERSONS]) for d in dirs])
    res = res.reshape(len(dirs), -1, 2)          # (seed, N, 2)
    shared = res.mean(0)                          # seed 평균 = 공유 성분
    dev = res - shared                            # seed마다 다른 부분
    # seed n개의 평균에는 seed 잡음이 1/n만큼 남아 있으므로, var(평균)을 그대로
    # '공유 성분'으로 읽으면 공유분을 σ²_seed/n만큼 **과대평가**한다. 실제로 이 추정치는
    # seed를 2 → 3 → 7개로 늘리는 동안 95.1% → 93.7% → 92.2%로 계속 내려갔다.
    # 아래 보정값이 n → ∞의 극한이고, "seed를 무한히 늘려도 남는 몫"은 그쪽이다.
    n = len(dirs)
    tab = []
    for k, axis in enumerate(("yaw", "pitch")):
        v_tot = res[:, :, k].var(ddof=1)
        v_sh = shared[:, k].var(ddof=1)
        v_dev = dev[:, :, k].var(ddof=1)
        v_seed = v_dev * n / (n - 1)              # seed 성분의 불편 추정
        v_sh_inf = max(v_sh - v_seed / n, 0.0)    # n → ∞ 극한의 공유 성분
        frac_inf = v_sh_inf / (v_sh_inf + v_seed)
        print(f"  {axis:<6} 전체 {v_tot:7.3f}  공유 {v_sh:7.3f} ({v_sh/v_tot:5.1%})  "
              f"seed마다 {v_dev:7.3f} ({v_dev/v_tot:5.1%})")
        print(f"         └ n→∞ 보정: 공유 {v_sh_inf:7.3f} ({frac_inf:5.1%})  "
              f"seed {v_seed:7.3f} ({1 - frac_inf:5.1%})")
        tab.append([axis, round(float(v_tot), 4), round(float(v_sh), 4),
                    round(float(v_dev), 4), round(float(v_sh / v_tot), 4),
                    round(float(v_sh_inf), 4), round(float(v_seed), 4),
                    round(float(frac_inf), 4)])
        put_numbers(**{f"a15_{axis}_shared_frac": round(float(v_sh / v_tot), 4),
                       f"a15_{axis}_seedvar_frac": round(float(v_dev / v_tot), 4),
                       f"a15_{axis}_shared_frac_inf": round(float(frac_inf), 4)})
    save_table("a15_variance_shared_vs_seed",
               ["axis", "total_var", "shared_var_n_seeds", "seed_var_raw",
                "shared_frac_n_seeds", "shared_var_inf", "seed_var_unbiased",
                "shared_frac_inf"], tab)
    put_numbers(a15_n_seeds=n)

    # 실제 ensemble 이득
    single = np.array([[angular_error(folds[d][p]["pred"], gaze[p]).mean()
                        for p in PERSONS] for d in dirs])
    ens = np.array([angular_error(np.mean([folds[d][p]["pred"] for d in dirs], 0),
                                  gaze[p]).mean() for p in PERSONS])
    r = paired(ens, single.mean(0))
    print(f"\nseed-ensemble: 단일 평균 {single.mean():.3f}° → {ens.mean():.3f}°  "
          f"Δ {r['delta']:+.3f}° ± {r['se']:.3f}  p={r['p']:.4f}")
    # 이론적 한계: 공유 성분만 남으면
    shared_err = angular_error(
        np.concatenate([np.mean([folds[d][p]["pred"] for d in dirs], 0)
                        for p in PERSONS]),
        np.concatenate([gaze[p] for p in PERSONS])).mean()
    print(f"  seed를 무한히 늘려도 공유 성분(n→∞ 보정 {tab[1][7]:.0%} of pitch 분산)은 남는다.")
    put_numbers(a15_single_mean=round(float(single.mean()), 4),
                a15_ens_mean=round(float(ens.mean()), 4),
                a15_ens_delta=round(float(r["delta"]), 4),
                a15_ens_p=sig(r["p"]))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax = axes[0]
    for k, axis in enumerate(("yaw", "pitch")):
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                ax.scatter(offs[i, :, k], offs[j, :, k], s=18,
                           c="#1f4e79" if k == 0 else "#c0504d",
                           label=f"{axis}" if (i, j) == (0, 1) else None)
    lim = [offs.min() * 1.1, offs.max() * 1.1]
    ax.plot(lim, lim, "--", lw=0.8, c="k")
    ax.set(xlabel="seed A의 사람 offset (°)", ylabel="seed B의 사람 offset (°)")
    ax.legend(fontsize=6)
    ax = axes[1]
    x = np.arange(2); w = 0.5
    sh = [t[4] for t in tab]
    ax.bar(x, sh, w, label="모든 seed가 공유 (ensemble로 못 줄임)", color="#1f4e79")
    ax.bar(x, [1 - v for v in sh], w, bottom=sh,
           label="seed마다 다름 (ensemble로 줄어듦)", color="#c6d9f0")
    ax.set_xticks(x, ["yaw", "pitch"]); ax.set(ylabel="잔차 분산 비중")
    ax.legend(fontsize=6)
    save_fig(fig, "28_ensemble_mechanism.png",
             {"axis": ["yaw", "pitch"], "shared_frac": sh})


if __name__ == "__main__":
    run()
