"""A20 — 이 과제에서 ablation 하나를 판정하려면 seed가 몇 개 필요한가.

A18·P5가 "항목당 1 seed로는 아무것도 말할 수 없다"를 보였다. 그러면 몇 개여야 하나.
답을 추측하지 않고 **기준 설정의 seed 7개로 직접 잰다**: seed들을 서로 겹치지 않는
두 묶음(n개씩)으로 갈라 각 묶음의 seed 평균끼리 15-fold 대응표본 차이를 만들고,
그 SE가 n에 따라 어떻게 줄어드는지 본다. 같은 설정끼리 갈랐으므로 참값 Δ는 0이고,
여기서 나오는 SE가 바로 '처리 효과를 재려 할 때 깔리는 잡음'이다.

그 SE로 검정력을 계산한다. 15 fold → df=14, 양측 α=0.05, 목표 검정력 80%.
CPU 전용, GPU 0.
"""
from __future__ import annotations

import glob
import itertools
import os

import numpy as np
from scipy import stats

from analysis.common import (PERSONS, load_run, plt, put_numbers, save_fig,
                             save_table, section)

BASE_GLOB = ("results/p1_seed*", "results/p4_seed*")
RUN_MINUTES = 29          # 15-fold 1회의 실측 소요 (results/JOBS.md)
ALPHA = 0.05
TARGET_POWER = 0.80


def required_ratio(n_folds: int = len(PERSONS), alpha: float = ALPHA,
                   power: float = TARGET_POWER) -> float:
    """|Δ|/SE 가 이 값 이상이어야 목표 검정력이 나온다 (대응표본 t, df = n-1)."""
    df = n_folds - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # 비중심 t의 검정력이 목표에 닿는 비중심도를 수치로 찾는다.
    lo, hi = 0.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        p = (1 - stats.nct.cdf(t_crit, df, mid)) + stats.nct.cdf(-t_crit, df, mid)
        if p < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run() -> None:
    section("A20 — ablation 판정에 필요한 seed 수 (경험적 검정력)")

    dirs = sorted(d for pat in BASE_GLOB for d in glob.glob(pat)
                  if os.path.exists(f"{d}/lopo.json")
                  and len(glob.glob(f"{d}/folds/*.npz")) >= len(PERSONS))
    errs = np.array([load_run(d)["fold_err"] for d in dirs])
    n_seeds = len(dirs)
    print(f"기준 설정 seed {n_seeds}개: {[os.path.basename(d) for d in dirs]}")

    ratio = required_ratio()
    print(f"\n대응표본 t (15 fold, df=14, 양측 α={ALPHA}), 목표 검정력 {TARGET_POWER:.0%}")
    print(f"  필요한 |Δ|/SE = {ratio:.3f}")

    # n개씩 겹치지 않게 갈라, seed 평균끼리의 차이의 SE를 모든 조합에서 잰다.
    print(f"\n{'묶음당 seed':>12}{'조합':>7}{'SE 중앙값':>11}"
          f"{'판정 가능 최소 |Δ|':>18}{'|Δ|=0.15 검정력':>16}")
    rows = []
    for n in range(1, n_seeds // 2 + 1):
        ses = []
        for a in itertools.combinations(range(n_seeds), n):
            rest = [i for i in range(n_seeds) if i not in a]
            for b in itertools.combinations(rest, n):
                d = errs[list(a)].mean(0) - errs[list(b)].mean(0)
                ses.append(d.std(ddof=1) / np.sqrt(len(d)))
        se = float(np.median(ses))
        mde = ratio * se                       # 최소 검출 가능 효과
        df = len(PERSONS) - 1
        t_crit = stats.t.ppf(1 - ALPHA / 2, df)
        nc = 0.15 / se
        pw = (1 - stats.nct.cdf(t_crit, df, nc)) + stats.nct.cdf(-t_crit, df, nc)
        print(f"{n:>12}{len(ses):>7}{se:>11.4f}{mde:>18.3f}°{pw:>15.0%}")
        rows.append([n, len(ses), round(se, 5), round(float(mde), 4), round(float(pw), 4)])

    save_table("a20_seed_power",
               ["seeds_per_arm", "n_splits", "se_median_deg",
                "min_detectable_delta_deg", "power_at_delta_0.15"], rows)

    # ---------------------------------------------------------------- 분산 모형
    # SE² ≈ c + k·(1/n_a + 1/n_b)  —  c는 seed로 지워지지 않는 바닥.
    # 위 표는 n_a = n_b = n인 대칭 분할이므로 SE² ≈ c + 2k/n이다. 최소제곱으로 맞춘다.
    ns = np.array([r[0] for r in rows], float)
    se2 = np.array([r[2] for r in rows], float) ** 2
    A = np.stack([np.ones_like(ns), 2.0 / ns], axis=1)
    (c, kk), *_ = np.linalg.lstsq(A, se2, rcond=None)
    pred = A @ np.array([c, kk])
    print(f"\n분산 모형 SE² = c + k·(1/n_a + 1/n_b)")
    print(f"  c = {c:.6f} (SE 바닥 {max(c, 0) ** .5:.4f}°), k = {kk:.6f}")
    print(f"  대칭 분할 적합: 관측 {np.round(np.sqrt(se2), 4)} vs 모형 "
          f"{np.round(np.sqrt(np.clip(pred, 0, None)), 4)}")

    def se_of(na, nb):
        return float(np.sqrt(max(c + kk * (1.0 / na + 1.0 / nb), 1e-12)))

    # 실제로 쓰는 설계는 비대칭이다: 변형 n_a개 vs 기준 7개. P5 실측으로 모형을 검증한다.
    print(f"\n모형 검증 — P5에서 실제로 잰 SE와 비교 (변형 n개 vs 기준 {n_seeds}개)")
    obs = [("풀링 2회", 3, 0.060), ("픽셀별 정규화", 2, 0.051), ("head pose 제거", 2, 0.066)]
    for name, na, se_obs in obs:
        print(f"  {name:<14} n={na}  실측 SE {se_obs:.3f}  모형 {se_of(na, n_seeds):.3f}")

    print(f"\n비대칭 설계의 비용 (기준은 seed {n_seeds}개 고정)")
    print(f"{'변형 seed':>10}{'GPU 시간':>10}{'SE(모형)':>10}{'판정 가능 최소 |Δ|':>18}")
    asym = []
    for na in (1, 2, 3, 5, 7):
        se_a = se_of(na, n_seeds)
        asym.append([na, round(na * RUN_MINUTES / 60, 2), round(se_a, 5),
                     round(ratio * se_a, 4)])
        print(f"{na:>10}{na * RUN_MINUTES / 60:>9.1f}h{se_a:>10.4f}{ratio * se_a:>17.3f}°")
    save_table("a20_asymmetric_cost",
               ["variant_seeds", "gpu_hours_1gpu", "se_model_deg",
                "min_detectable_delta_deg"], asym)
    put_numbers(a20_model_c=round(float(c), 6), a20_model_k=round(float(kk), 6),
                a20_se_floor=round(float(max(c, 0) ** .5), 5),
                a20_mde_floor=round(float(ratio * max(c, 0) ** .5), 4),
                a20_asym_mde_3_seeds=asym[2][3])

    need = next((r[0] for r in rows if r[4] >= TARGET_POWER), None)
    best = rows[-1]
    if need is None:
        print(f"\n→ seed {best[0]}개씩으로도 |Δ|=0.15°는 검정력 {best[4]:.0%}에 그친다. "
              f"그 설정에서 판정 가능한 최소 Δ는 {best[3]:.3f}°다.")
        # SE ∝ 1/sqrt(n)가 아니라 공유 성분 때문에 바닥이 있다 — 외삽은 하지 않는다.
    else:
        print(f"\n→ 묶음당 seed {need}개면 |Δ|=0.15°를 검정력 {TARGET_POWER:.0%}로 잡는다.")

    put_numbers(a20_required_t_ratio=round(float(ratio), 4),
                a20_se_1_seed=rows[0][2], a20_mde_1_seed=rows[0][3],
                a20_se_max_arm=best[2], a20_mde_max_arm=best[3],
                a20_seeds_per_arm_max=best[0],
                a20_power_at_015_max_arm=best[4],
                a20_seeds_needed_for_015=need)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    xs = [r[0] for r in rows]
    ax.plot(xs, [r[3] for r in rows], "o-", color="#1f4e79",
            label="판정 가능한 최소 |Δ| (검정력 80%)")
    ax.axhline(0.15, ls="--", lw=0.9, c="#c0504d", label="ablation Δ의 전형적 크기 0.15°")
    for r in rows:
        ax.annotate(f"{r[3]:.2f}°", (r[0], r[3]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)
    ax.set(xlabel="묶음당 seed 수 (양쪽 각각)", ylabel="최소 검출 가능 Δ (°)",
           xticks=xs, ylim=(0, max(r[3] for r in rows) * 1.25))
    ax.legend(fontsize=7)
    save_fig(fig, "33_seed_power.png",
             {"seeds_per_arm": xs, "min_detectable_delta_deg": [r[3] for r in rows],
              "se_median_deg": [r[2] for r in rows]})

    print(f"""
결론: ablation 하나를 판정하는 비용을 숫자로 적을 수 있다.

  (1) **1차 ablation 설계는 처음부터 답이 나올 수 없었다.** 양쪽 seed 1개씩이면
      판정 가능한 최소 |Δ|가 {rows[0][3]:.3f}°다. 관측된 ablation Δ는 전부 0.25° 이하였다.
      p=0.0113이 나온 풀링 2회조차 이 기준으로는 "운이 좋았던 쪽"이다.
  (2) 실제로 쓴 비대칭 설계(변형 n개 vs 기준 {n_seeds}개)는 더 낫다. 모형이 P5에서
      실측한 SE 세 개를 재현하므로 (위 검증) 비용을 예측할 수 있다 —
      변형 seed 3개면 최소 검출 가능 Δ가 {asym[2][3]:.3f}°, GPU 1.5시간이다.
  (3) **SE는 1/√n로 줄지 않는다.** 잔차의 9할이 seed 간에 공유되므로(A15) 바닥이 있다.
      모형이 주는 바닥은 SE {max(c, 0) ** .5:.3f}° → 최소 검출 가능 Δ {ratio * max(c, 0) ** .5:.3f}°이지만,
      **적합점이 3개뿐이라 이 외삽은 약하다.** 자리수 이야기로만 읽어야 한다.

  실행 지침: 항목을 줄이고 seed를 늘린다. 같은 GPU 시간이면 8종×1 seed
  (최소 검출 {rows[0][3]:.2f}°)보다 3종×3 seed(최소 검출 {asym[2][3]:.2f}°)가 답할 수 있는 질문이 많다.""")


if __name__ == "__main__":
    run()
