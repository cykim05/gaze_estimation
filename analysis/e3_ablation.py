"""E3 — ablation 표: 항목 | EDA의 예측 | Δ ± SE | p | 일치 여부 | 흡수 여부.

PREREG 4·5절. 기준은 P1 seed 0(같은 seed로 돌린 변형과 짝을 맞춘다).
흡수 기준: 15-fold 대응표본 양측 p < 0.05로 기준보다 **우세**.
"""
from __future__ import annotations

import os

import numpy as np

from analysis.common import (sig, PERSONS, angular_error, load_folds, load_run, paired,
                             plt, put_numbers, save_table, save_fig, section)

# PREREG 4절에 적은 사전 예상
PREDICTION = {
    "ab_nohead":  ("head pose 제거", "거의 차이 없음 (Δ<0.2°, 구분 불가 가능성)", "none"),
    "ab_pixnorm": ("픽셀별 정규화", "악화 (Δ>+0.15°)", "worse"),
    "ab_angular": ("angular(cos) 손실", "소폭 개선 또는 구분 불가 (≤0.15°)", "better_or_none"),
    "ab_l1":      ("L1 손실", "중앙값 개선, 평균은 구분 불가", "none"),
    "ab_k5":      ("3×3 → 5×5 커널", "구분 불가 또는 소폭 악화", "none_or_worse"),
    "ab_pool2":   ("풀링 3회 → 2회", "악화", "worse"),
}
BASE = "results/p1_seed0"

# P5 (PREREG 4·5절) — "기준 대비 0.15° 이상 차이 난 항목에 seed 1~2개 추가".
# 단일 seed 비교는 seed 효과를 처리 효과로 집계한다(A18). 여기서는 **seed 평균끼리**
# 짝을 맞춰 다시 본다. 흡수 여부는 P5가 돌아간 항목에 대해서는 이 표로 정한다.
SEED_RUNS = {
    "base":       ["p1_seed0", "p1_seed1", "p1_seed2",
                   "p4_seed3", "p4_seed4", "p4_seed5", "p4_seed6"],
    "ab_pool2":   ["ab_pool2", "ab_pool2_s1", "ab_pool2_s2"],
    "ab_pixnorm": ["ab_pixnorm", "ab_pixnorm_s1"],
    "ab_nohead":  ["ab_nohead", "ab_nohead_s1"],
}


def fold_errors(run_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not os.path.exists(os.path.join(run_dir, "lopo.json")):
        return None
    load_run(run_dir)
    folds = load_folds(run_dir, keys=["pred", "gaze"])
    if len(folds) < len(PERSONS):
        return None
    errs = np.array([angular_error(folds[p]["pred"], folds[p]["gaze"]).mean()
                     for p in PERSONS])
    med = np.array([np.median(angular_error(folds[p]["pred"], folds[p]["gaze"]))
                    for p in PERSONS])
    return errs, med


def seed_mean_errors(runs: list[str]) -> np.ndarray | None:
    """여러 seed 실행의 fold별 오차를 쌓아 (seed, 15) 행렬로."""
    got = [fold_errors(f"results/{r}")[0] for r in runs
           if fold_errors(f"results/{r}") is not None]
    return np.array(got) if got else None


def multiseed_stage() -> dict:
    """P5: seed 평균끼리 다시 비교한다. 반환 = {key: paired 결과}."""
    base = seed_mean_errors(SEED_RUNS["base"])
    if base is None or len(base) < 2:
        print("\nP5: 기준 seed가 2개 미만이다. 건너뛴다.")
        return {}
    print(f"\n{'-' * 76}\nP5 (PREREG 4·5절) — seed 평균끼리 다시 비교")
    print(f"기준 seed {len(base)}개, 평균 {base.mean():.3f}°, "
          f"seed 간 SD {base.mean(1).std(ddof=1):.3f}°")
    print(f"\n{'항목':<18}{'seed':>5}{'평균':>8}{'Δ(seed 평균)':>14}{'SE':>8}{'p':>9}"
          f"{'Δ(seed0 단독)':>15}  판정")
    out, rows = {}, []
    for key, runs in SEED_RUNS.items():
        if key == "base":
            continue
        var = seed_mean_errors(runs)
        if var is None or len(var) < 2:
            print(f"{PREDICTION[key][0]:<18}{0 if var is None else len(var):>5}"
                  f"  (seed 부족 — 단일 seed 결과를 그대로 둔다)")
            continue
        r = paired(var.mean(0), base.mean(0))
        single = var[0].mean() - base[0].mean()
        v = verdict(r["delta"], r["p"])
        print(f"{PREDICTION[key][0]:<18}{len(var):>5}{var.mean():>8.3f}"
              f"{r['delta']:>+14.3f}{r['se']:>8.3f}{r['p']:>9.4f}{single:>+15.3f}  {v}")
        rows.append([PREDICTION[key][0], len(var), round(float(var.mean()), 4),
                     round(float(r["delta"]), 4), round(float(r["se"]), 4),
                     sig(r["p"]), sig(r["wilcoxon_p"]),
                     round(float(single), 4), v])
        put_numbers(**{f"e3p5_{key}_n_seeds": len(var),
                       f"e3p5_{key}_mean": round(float(var.mean()), 4),
                       f"e3p5_{key}_delta": round(float(r["delta"]), 4),
                       f"e3p5_{key}_se": round(float(r["se"]), 4),
                       f"e3p5_{key}_p": sig(r["p"]),
                       f"e3p5_{key}_delta_single_seed": round(float(single), 4)})
        out[key] = r
    if len(rows) >= 2:
        # 단일 seed Δ가 전부 음수였다가 전부 0 근처로 모인다면, 원인은 각 변형이
        # 아니라 **기준 seed 하나**다. 그 몫을 직접 계산한다.
        d_single = np.array([r[7] for r in rows])
        d_multi = np.array([r[3] for r in rows])
        bias = base[0].mean() - base.mean()
        print(f"\n단일 seed Δ 평균 {d_single.mean():+.3f}°  →  "
              f"seed 평균 Δ 평균 {d_multi.mean():+.3f}°  (이동량 {d_multi.mean() - d_single.mean():+.3f}°)")
        print(f"기준으로 쓴 seed 0은 7개 seed 평균보다 {bias:+.3f}° 나쁘다 "
              f"(7개 중 {sorted(base.mean(1), reverse=True).index(base[0].mean()) + 1}번째로 나쁜 쪽).")
        print(f"  → 이동량 {d_multi.mean() - d_single.mean():+.3f}°의 "
              f"{abs(bias) / abs(d_multi.mean() - d_single.mean()) * 100:.0f}%가 그 한 가지로 설명된다. "
              f"변형이 좋아 보였던 게 아니라 기준이 운이 나빴던 것이다.")
        put_numbers(e3p5_delta_single_mean=round(float(d_single.mean()), 4),
                    e3p5_delta_multi_mean=round(float(d_multi.mean()), 4),
                    e3p5_base_seed0_bias=round(float(bias), 4))

    if rows:
        save_table("e3_ablation_multiseed",
                   ["item", "n_seeds", "mean_deg", "delta_seedmean", "se", "p",
                    "wilcoxon_p", "delta_single_seed", "verdict"], rows)
        put_numbers(e3p5_base_n_seeds=len(base),
                    e3p5_base_mean=round(float(base.mean()), 4))
    return out


def verdict(delta: float, p: float) -> str:
    if p >= 0.05:
        return "구분 불가"
    return "유의하게 악화" if delta > 0 else "유의하게 개선"


def matches(expect: str, delta: float, p: float) -> str:
    is_sig = p < 0.05
    got = "none" if not is_sig else ("worse" if delta > 0 else "better")
    if expect == "none":
        return "일치" if got == "none" else "어긋남"
    if expect == "worse":
        return "일치" if got == "worse" else "어긋남"
    if expect == "better_or_none":
        return "일치" if got in ("none", "better") else "어긋남"
    if expect == "none_or_worse":
        return "일치" if got in ("none", "worse") else "어긋남"
    return "?"


def run():
    section("E3 — ablation")
    base = fold_errors(BASE)
    if base is None:
        print(f"  기준 {BASE}가 아직 없다.")
        return
    base_err, base_med = base
    print(f"기준 {BASE}: 평균 {base_err.mean():.3f}°, 중앙값 평균 {base_med.mean():.3f}°")

    print(f"\n{'항목':<20}{'평균':>8}{'Δ':>9}{'SE':>8}{'p':>9}  {'판정':<14}{'예측 일치':<10}")
    rows, absorbed = [], []
    for key, (label, pred_text, expect) in PREDICTION.items():
        got = fold_errors(f"results/{key}")
        if got is None:
            print(f"{label:<20}{'—':>8}  (결과 없음)")
            rows.append([label, pred_text, None, None, None, "결과 없음", "—", "—"])
            continue
        errs, med = got
        r = paired(errs, base_err)
        v = verdict(r["delta"], r["p"])
        agree = matches(expect, r["delta"], r["p"])
        print(f"{label:<20}{errs.mean():>8.3f}{r['delta']:>+9.3f}{r['se']:>8.3f}"
              f"{r['p']:>9.4f}  {v:<14}{agree:<10}")
        take = r["p"] < 0.05 and r["delta"] < 0
        if take:
            absorbed.append((label, r))
        rows.append([label, pred_text, round(float(errs.mean()), 4),
                     round(float(r["delta"]), 4), round(float(r["se"]), 4),
                     sig(r["p"]), v, agree,
                     "흡수" if take else ("유지" if r["p"] >= 0.05 else "기각")])
        # 중앙값도 (L1 예측 확인용)
        rm = paired(med, base_med)
        put_numbers(**{
            f"e3_{key}_mean": round(float(errs.mean()), 4),
            f"e3_{key}_delta": round(float(r["delta"]), 4),
            f"e3_{key}_se": round(float(r["se"]), 4),
            f"e3_{key}_p": sig(r["p"]),
            f"e3_{key}_median_delta": round(float(rm["delta"]), 4),
            f"e3_{key}_median_p": sig(rm["p"]),
        })

    save_table("e3_ablation",
               ["item", "eda_prediction", "mean_deg", "delta_vs_base", "se", "p",
                "verdict", "prediction_matched", "absorbed"], rows)

    p5 = multiseed_stage()

    # ------------------------------------------------------------ 흡수 결정
    print(f"\n{'-' * 76}\n흡수 기준(PREREG 5절): 15-fold 대응표본 양측 p < 0.05로 기준보다 우세.")
    print("P5가 돌아간 항목은 **seed 평균끼리의 비교**로 정한다(단일 seed 비교는 A18이")
    print("보인 대로 seed 효과를 처리 효과로 집계한다). 규칙은 실험 전에 적은 것이고,")
    print("결과를 본 뒤 바꾸지 않았다.")

    final, overturned = [], []
    for label, r in absorbed:
        key = next(k for k, v in PREDICTION.items() if v[0] == label)
        if key in p5:
            if p5[key]["p"] < 0.05 and p5[key]["delta"] < 0:
                final.append((label, p5[key], f"P5 seed {len(SEED_RUNS[key])}개"))
            else:
                overturned.append((label, r, p5[key]))
        else:
            final.append((label, r, "단일 seed (P5 미적용)"))

    for label, single, multi in overturned:
        print(f"\n  [뒤집힘] {label}")
        print(f"      단일 seed : Δ {single['delta']:+.3f}° ± {single['se']:.3f}, "
              f"p={single['p']:.4f}  → 유의하게 개선")
        print(f"      seed 평균 : Δ {multi['delta']:+.3f}° ± {multi['se']:.3f}, "
              f"p={multi['p']:.4f}  → 구분 불가 "
              f"(|Δ|가 {abs(single['delta']) / max(abs(multi['delta']), 1e-9):.0f}배 줄었다)")
        print(f"      → 절대 규칙 3: 사전 기준 미만족 → 더 단순한 쪽(기준)을 유지한다.")

    if final:
        best = min(final, key=lambda t: t[1]["delta"])
        print(f"\n  → 흡수: {best[0]} (Δ {best[1]['delta']:+.3f}°, p={best[1]['p']:.4f}, "
              f"근거 {best[2]})")
        if len(final) > 1:
            print(f"  → 나머지 {len(final)-1}개는 '추가 검증 필요'로 남긴다 "
                  "(조합 효과를 측정하지 않았다)")
        put_numbers(e3_absorbed=best[0], e3_absorbed_delta=round(float(best[1]["delta"]), 4))
    else:
        print("\n  → 기준을 넘은 변형이 없다. 더 단순한 쪽(기준 모델)을 유지한다.")
        put_numbers(e3_absorbed=None)
    put_numbers(e3_overturned_by_p5=[o[0] for o in overturned])

    done = [r for r in rows if r[2] is not None]
    if done:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        y = np.arange(len(done))
        deltas = [r[3] for r in done]
        ses = [r[4] for r in done]
        colors = ["#c0504d" if r[5] < 0.05 and r[3] > 0 else
                  ("#4f81bd" if r[5] < 0.05 else "#9e9e9e") for r in done]
        ax.barh(y, deltas, xerr=ses, color=colors, height=0.6, capsize=3)
        ax.axvline(0, lw=0.8, c="k")
        ax.set_yticks(y, [r[0] for r in done], fontsize=7)
        ax.set(xlabel="기준 대비 LOPO 오차 변화 Δ (°, 음수가 개선)")
        ax.invert_yaxis()
        save_fig(fig, "21_ablation.png",
                 {"item": [r[0] for r in done], "delta": deltas, "se": ses,
                  "p": [r[5] for r in done]})
    return rows


if __name__ == "__main__":
    run()
