"""E3 — ablation 표: 항목 | EDA의 예측 | Δ ± SE | p | 일치 여부 | 흡수 여부.

PREREG 4·5절. 기준은 P1 seed 0(같은 seed로 돌린 변형과 짝을 맞춘다).
흡수 기준: 15-fold 대응표본 양측 p < 0.05로 기준보다 **우세**.
"""
from __future__ import annotations

import os

import numpy as np

from analysis.common import (PERSONS, angular_error, load_folds, load_run, paired,
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


def verdict(delta: float, p: float) -> str:
    if p >= 0.05:
        return "구분 불가"
    return "유의하게 악화" if delta > 0 else "유의하게 개선"


def matches(expect: str, delta: float, p: float) -> str:
    sig = p < 0.05
    got = "none" if not sig else ("worse" if delta > 0 else "better")
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
                     round(float(r["p"]), 5), v, agree,
                     "흡수" if take else ("유지" if r["p"] >= 0.05 else "기각")])
        # 중앙값도 (L1 예측 확인용)
        rm = paired(med, base_med)
        put_numbers(**{
            f"e3_{key}_mean": round(float(errs.mean()), 4),
            f"e3_{key}_delta": round(float(r["delta"]), 4),
            f"e3_{key}_se": round(float(r["se"]), 4),
            f"e3_{key}_p": round(float(r["p"]), 5),
            f"e3_{key}_median_delta": round(float(rm["delta"]), 4),
            f"e3_{key}_median_p": round(float(rm["p"]), 5),
        })

    save_table("e3_ablation",
               ["item", "eda_prediction", "mean_deg", "delta_vs_base", "se", "p",
                "verdict", "prediction_matched", "absorbed"], rows)

    print(f"\n흡수 기준(PREREG 5절): 15-fold 대응표본 양측 p < 0.05로 기준보다 우세")
    if absorbed:
        best = min(absorbed, key=lambda t: t[1]["delta"])
        print(f"  → 흡수: {best[0]} (Δ {best[1]['delta']:+.3f}°, p={best[1]['p']:.4f})")
        if len(absorbed) > 1:
            print(f"  → 나머지 {len(absorbed)-1}개는 '추가 검증 필요'로 남긴다 "
                  "(조합 효과를 측정하지 않았다)")
        put_numbers(e3_absorbed=best[0], e3_absorbed_delta=round(float(best[1]["delta"]), 4))
    else:
        print("  → 기준을 넘은 변형이 없다. 더 단순한 쪽(기준 모델)을 유지한다.")
        put_numbers(e3_absorbed=None)

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
