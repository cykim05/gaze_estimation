"""최종 결과 표 — baseline들 + 이전 실행 + 최종 모델(단일/ensemble)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from analysis.common import (PERSONS, angular_error, load_folds, load_run, paired,
                             put_numbers, save_table, section)

RIDGE = [5.31, 7.76, 6.55, 8.05, 6.40, 7.35, 7.59, 7.97, 7.89, 8.51, 6.93, 6.63,
         7.07, 7.85, 7.20]
MEANPRED = [8.43, 8.63, 9.00, 9.60, 8.97, 9.73, 10.34, 9.49, 9.61, 8.97, 8.66, 8.43,
            8.60, 9.00, 10.75]


def json_folds(path):
    if not os.path.exists(path):
        return None
    meta = json.load(open(path, encoding="utf-8"))
    g = {r["test_person"]: r["test_error"] for r in meta["results"]}
    return np.array([g[p] for p in PERSONS]) if len(g) == len(PERSONS) else None


def run():
    section("최종 결과 표")
    rows = [["평균 gaze 예측 (하한선)"] + MEANPRED + [round(float(np.mean(MEANPRED)), 3)],
            ["ridge 회귀 (저장소 baseline)"] + RIDGE + [round(float(np.mean(RIDGE)), 3)]]
    base_ref = None

    for label, path in [("이전 실행 A (b128, do0.2, 15ep)", "results/base/lopo.json"),
                        ("이전 실행 B (b128, do0.5, 20ep)", "results/final/lopo.json")]:
        v = json_folds(path)
        if v is not None:
            rows.append([label] + [round(float(x), 3) for x in v]
                        + [round(float(v.mean()), 3)])

    seeds = sorted(glob.glob("results/p1_seed*"))
    seed_errs = []
    for sd in seeds:
        v = json_folds(os.path.join(sd, "lopo.json"))
        if v is not None:
            seed_errs.append(v)
    if seed_errs:
        A = np.array(seed_errs)
        mean_single = A.mean(0)
        rows.append([f"최종 모델 단일 (seed {len(seed_errs)}개 평균)"]
                    + [round(float(x), 3) for x in mean_single]
                    + [round(float(mean_single.mean()), 3)])
        base_ref = mean_single
        put_numbers(final_single_mean=round(float(mean_single.mean()), 4),
                    final_single_seed_means=[round(float(v.mean()), 4) for v in A],
                    final_n_seeds=len(seed_errs))

        # seed-ensemble
        preds, gazes = {}, {}
        ok = True
        for sd in seeds:
            f = load_folds(sd, keys=["pred", "gaze"])
            if len(f) < len(PERSONS):
                ok = False; break
            for p in PERSONS:
                preds.setdefault(p, []).append(f[p]["pred"])
                gazes[p] = f[p]["gaze"]
        if ok and len(seeds) > 1:
            ens = np.array([angular_error(np.mean(preds[p], 0), gazes[p]).mean()
                            for p in PERSONS])
            r = paired(ens, mean_single)
            rows.append([f"최종 모델 seed-ensemble ({len(seeds)})"]
                        + [round(float(x), 3) for x in ens]
                        + [round(float(ens.mean()), 3)])
            print(f"seed-ensemble vs 단일: Δ {r['delta']:+.3f}° ± {r['se']:.3f}, "
                  f"p = {r['p']:.4f} → {'유의' if r['significant'] else '구분 불가'}")
            put_numbers(final_seed_ens_mean=round(float(ens.mean()), 4),
                        final_seed_ens_delta=round(float(r["delta"]), 4),
                        final_seed_ens_p=round(float(r["p"]), 5))

    save_table("final_results", ["method"] + list(PERSONS) + ["mean_deg"], rows)
    print()
    for r in rows:
        print(f"  {r[0]:<38}{r[-1]:>8}")

    if base_ref is not None:
        for label, v in [("ridge baseline", np.array(RIDGE)),
                         ("하한선", np.array(MEANPRED))]:
            rr = paired(base_ref, v)
            print(f"\n최종 모델 vs {label}: Δ {rr['delta']:+.3f}° ± {rr['se']:.3f}, "
                  f"t = {rr['t']:.2f}, p = {rr['p']:.2e}")
            put_numbers(**{f"final_vs_{label.replace(' ', '_')}_delta":
                           round(float(rr["delta"]), 4),
                           f"final_vs_{label.replace(' ', '_')}_p": float(f"{rr['p']:.3e}")})


if __name__ == "__main__":
    run()
