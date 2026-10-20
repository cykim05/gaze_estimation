"""최종 결과 표 — baseline들 + 이전 실행 + 최종 모델(단일/ensemble)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from analysis.common import (sig, PERSONS, angular_error, load_folds, load_run, paired,
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

    # ---------------------------------------------------------- 수집
    # (label, fold 오차, 기준과 같은 설정인가)
    series = [("평균 gaze 예측 (하한선)", np.array(MEANPRED, float), False),
              ("ridge 회귀 (저장소 baseline)", np.array(RIDGE, float), False)]
    for label, path, same in [
            ("이전 실행 A (b128, do0.2, 15ep)", "results/base/lopo.json", False),
            # runB는 최종 모델과 **같은 설정**이고, p1_seed0과 15 fold 전부
            # 비트 단위로 같다(최대 차이 0.000°). 따라서 아래 '단일(seed 평균)'의
            # 구성원 하나다. 다른 방법이 아니다.
            ("이전 실행 B (= P1 seed 0, 최종과 같은 설정 · seed 1개)",
             "results/final/lopo.json", True)]:
        v = json_folds(path)
        if v is not None:
            series.append((label, v, same))

    # 아직 돌고 있는 실행은 lopo.json이 없다. 완주한 것만 쓴다.
    seeds, seed_errs = [], []
    for sd in sorted(glob.glob("results/p1_seed*")) + sorted(glob.glob("results/p4_seed*")):
        v = json_folds(os.path.join(sd, "lopo.json"))
        if v is not None and len(glob.glob(f"{sd}/folds/*.npz")) >= len(PERSONS):
            seeds.append(sd); seed_errs.append(v)

    ref = None
    if seed_errs:
        A = np.array(seed_errs)
        ref = A.mean(0)
        series.append((f"최종 모델 단일 (seed {len(seeds)}개 평균)", ref, True))
        put_numbers(final_single_mean=round(float(ref.mean()), 4),
                    final_single_seed_means=[round(float(v.mean()), 4) for v in A],
                    final_n_seeds=len(seeds))

        # seed-ensemble: 예측을 평균
        preds, gazes, ok = {}, {}, True
        for sd in seeds:
            f = load_folds(sd, keys=["pred", "gaze"])
            if len(f) < len(PERSONS):
                ok = False; break
            for pp in PERSONS:
                preds.setdefault(pp, []).append(f[pp]["pred"])
                gazes[pp] = f[pp]["gaze"]
        if ok and len(seeds) > 1:
            ens = np.array([angular_error(np.mean(preds[pp], 0), gazes[pp]).mean()
                            for pp in PERSONS])
            series.append((f"  + seed-ensemble ({len(seeds)})", ens, False))
            r = paired(ens, ref)
            put_numbers(final_seed_ens_mean=round(float(ens.mean()), 4),
                        final_seed_ens_delta=round(float(r["delta"]), 4),
                        final_seed_ens_p=sig(r["p"]))

            # + epoch-ensemble. E2와 같은 코드 경로를 쓴다(K는 개발용 fold의 검증으로만).
            from analysis.e2_ensemble import collect, fold_err, pick_k_on_dev
            k, _ = pick_k_on_dev(seeds)
            data = collect(seeds, k)
            if len(data) == len(seeds):
                g0 = data[seeds[0]]["gaze"]
                both = fold_err({pp: np.mean([data[d]["epoch_ens"][pp] for d in seeds], 0)
                                 for pp in PERSONS}, g0)
                series.append((f"  + seed-ens + epoch-ens (K={k})  ← 최종", both, False))
                rb = paired(both, ref)
                put_numbers(final_model_mean=round(float(both.mean()), 4),
                            final_model_delta_vs_single=round(float(rb["delta"]), 4),
                            final_model_p=sig(rb["p"]),
                            final_model_epoch_K=k)

    # ---------------------------------------------------------- 표
    # Δ·t·p는 모두 '최종 모델 단일(seed 평균)'을 기준으로 한 15-fold 대응표본이다.
    rows = []
    for label, v, same in series:
        row = [label] + [round(float(x), 3) for x in v] + [round(float(v.mean()), 4)]
        if ref is None or np.array_equal(v, ref):
            row += [0.0, 0.0, None, None, "기준"]
        else:
            r = paired(v, ref)
            if same:
                # 기준과 같은 설정이고 기준 평균에 그 자신이 들어 있다. 검정이 아니다 —
                # 이 줄의 Δ는 'seed 추첨의 폭'을 보여 주는 값으로만 읽는다.
                verdict = "검정 아님 (같은 설정 · 기준에 포함)"
            else:
                verdict = "유의" if r["significant"] else "구분 불가"
            row += [round(float(r["delta"]), 4), round(float(r["se"]), 4),
                    round(float(r["t"]), 3), sig(r["p"]), verdict]
        rows.append(row)
    save_table("final_results",
               ["method"] + list(PERSONS)
               + ["mean_deg", "delta_vs_final_single", "se", "t", "p", "verdict"], rows)

    print(f"\n{'방법':<48}{'평균':>8}{'Δ vs 단일':>11}{'p':>11}  판정")
    for row in rows:
        pv = "—" if row[-2] is None else f"{row[-2]:.2e}"
        print(f"  {row[0]:<46}{row[16]:>8.3f}{row[17]:>+11.3f}{pv:>11}  {row[-1]}")
    print("\n'이전 실행 B'는 최종 모델과 같은 설정이고 p1_seed0과 15 fold 전부 같다"
          "(최대 차이 0.000°).")
    print("즉 '단일(seed 평균)'의 구성원이다. 그 줄의 Δ +0.186°는 설정의 차이가 아니라"
          " **seed 추첨의 폭**이고,")
    print("그래서 p값에 유의 판정을 붙이지 않았다(기준 평균이 그 실행을 포함한다).")

    if ref is not None:
        for label, v in [("ridge baseline", np.array(RIDGE, float)),
                         ("하한선", np.array(MEANPRED, float))]:
            rr = paired(ref, v)
            print(f"\n최종 모델 단일 vs {label}: Δ {rr['delta']:+.3f}° ± {rr['se']:.3f}, "
                  f"t = {rr['t']:.2f}, p = {rr['p']:.2e}")
            key = label.replace(" ", "_")
            put_numbers(**{f"final_vs_{key}_delta": round(float(rr["delta"]), 4),
                           f"final_vs_{key}_p": sig(rr["p"])})


if __name__ == "__main__":
    run()
