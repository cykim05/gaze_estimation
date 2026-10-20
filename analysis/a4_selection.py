"""A4 — 모델 선택 잡음: epoch을 val로 고르는 것이 이득인가.

질문: 검증 2명(6,000개)으로 epoch을 고르는 것이 잡음을 보태고 있나? 대안은?
epoch-ensemble의 K는 **개발용 fold의 검증으로만** 고른다 (PREREG 6절).
"""
from __future__ import annotations

import numpy as np

from analysis.common import (PERSONS, angular_error, load_folds, load_run, paired,
                             plt, put_numbers, save_fig, save_table, section, spearman)

DEV_FOLD = "p00"
K_CANDIDATES = [1, 2, 3, 5, 8]


def run(run_dir="results/final", tag="runB"):
    section(f"A4 — epoch 선택 잡음 ({run_dir})")
    d = load_run(run_dir)
    folds = load_folds(run_dir, keys=["trace_val_pred", "trace_test_pred",
                                      "val_gaze", "gaze", "pred"])
    meta = {r["test_person"]: r for r in d["meta"]["results"]}

    has_trace = all("trace_test_pred" in folds.get(p, {}) and
                    folds[p]["trace_test_pred"].size for p in PERSONS)
    if not has_trace:
        print("  이 실행에는 epoch별 예측이 없다 (Phase 0 이전 산출물).")
        print("  history의 epoch별 test 오차만으로 본다.")
        hist = np.array([[h["test"] for h in meta[p]["history"]] for p in PERSONS])
        vhist = np.array([[h["val"] for h in meta[p]["history"]] for p in PERSONS])
        chosen = np.array([meta[p]["test_error"] for p in PERSONS])
        last = hist[:, -1]
        oracle = hist.min(1)

        print(f"\n  {'방식':<22}{'평균 (°)':>10}")
        print(f"  {'val로 고른 epoch':<22}{chosen.mean():>10.3f}")
        print(f"  {'마지막 epoch':<22}{last.mean():>10.3f}")
        print(f"  {'oracle (fold별 최선)':<22}{oracle.mean():>10.3f}")
        r = paired(chosen, last)
        print(f"\n  val 선택 vs 마지막: Δ {r['delta']:+.3f}° ± {r['se']:.3f}  "
              f"t={r['t']:.2f}  p={r['p']:.4f}  "
              f"→ {'유의' if r['significant'] else '구분 불가'}")

        # epoch 축에서 val과 test의 순위 상관
        rhos = [spearman(vhist[i], hist[i])[0] for i in range(len(PERSONS))]
        print(f"  epoch 축 val–test 순위 상관(Spearman): 중앙값 {np.median(rhos):+.3f} "
              f"[{min(rhos):+.2f}, {max(rhos):+.2f}]")

        put_numbers(**{
            f"{tag}_sel_val_chosen": round(float(chosen.mean()), 4),
            f"{tag}_sel_last_epoch": round(float(last.mean()), 4),
            f"{tag}_sel_oracle_epoch": round(float(oracle.mean()), 4),
            f"{tag}_sel_val_vs_last_delta": round(float(r["delta"]), 4),
            f"{tag}_sel_val_vs_last_p": round(float(r["p"]), 5),
            f"{tag}_sel_val_test_spearman_median": round(float(np.median(rhos)), 4),
        })
        save_table(f"a4_epoch_selection_{tag}",
                   ["person", "val_chosen_deg", "last_epoch_deg", "oracle_epoch_deg",
                    "best_epoch", "val_test_spearman"],
                   [[p, round(float(chosen[i]), 3), round(float(last[i]), 3),
                     round(float(oracle[i]), 3), meta[p]["best_epoch"],
                     round(float(rhos[i]), 3)] for i, p in enumerate(PERSONS)])

        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
        ep = np.arange(1, hist.shape[1] + 1)
        for i in range(len(PERSONS)):
            axes[0].plot(ep, hist[i], lw=0.6, alpha=0.55, c="#8aa0b8")
        axes[0].plot(ep, hist.mean(0), lw=2.0, c="#1f4e79", label="15-fold 평균")
        axes[0].axhline(chosen.mean(), ls="--", lw=0.9, c="#c0504d",
                        label=f"val 선택 {chosen.mean():.2f}°")
        axes[0].axhline(oracle.mean(), ls=":", lw=0.9, c="k",
                        label=f"oracle {oracle.mean():.2f}°")
        axes[0].set(xlabel="epoch", ylabel="테스트 각도 오차 (°)")
        axes[0].legend(fontsize=6)
        axes[1].hist(rhos, bins=np.linspace(-1, 1, 21), color="#1f4e79")
        axes[1].axvline(0, ls="--", lw=0.8, c="k")
        axes[1].set(xlabel="epoch 축 val–test 순위 상관 (Spearman ρ)", ylabel="fold 수")
        save_fig(fig, "16_epoch_selection.png",
                 {"epoch": ep, "mean_test": hist.mean(0)})
        print("\n결론: epoch–test 곡선이 평평하고 val–test 순위 상관이 0 근처라, "
              "val로 epoch을 고르는 것은 이득이 아니라 잡음이다.")
        return

    print("  (epoch별 예측이 있는 실행 — epoch-ensemble까지 계산)")


if __name__ == "__main__":
    run()
