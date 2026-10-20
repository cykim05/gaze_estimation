"""A16 — 라벨 없이 어려운 샘플을 알아낼 수 있나. **[진단용]** (자율 탐색)

실용적 질문: 시스템이 "이 예측은 못 믿겠다"고 말할 수 있나?
라벨 없이 쓸 수 있는 신호로 좌·우눈 불일치가 있다(A10). 그 크기로 샘플을 줄 세우고,
상위 x%를 버렸을 때 남은 샘플의 오차가 얼마나 내려가는지 본다.

비교 기준:
  - 무작위로 버리기 (아무 정보 없음)
  - 실제 오차로 버리기 (오라클, 라벨 필요)
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, angular_error, load_run, pearson, plt,
                             put_numbers, save_fig, save_table, section, spearman)

FLIP = np.array([-1.0, 1.0], dtype=np.float32)
DROP = [0, 5, 10, 20, 30, 50]


def run(run_dir="results/final", tag="runB"):
    section(f"A16 — 라벨 없는 신뢰도 신호 [진단용] ({run_dir})")
    import dataset
    d = load_run(run_dir)
    df = dataset.label_frame()
    person, pred, gaze = d["person"], d["pred"], d["gaze"]
    eye = df["which_eye"].to_numpy()
    key = (df["person"].astype(str) + "|" + df["day"].astype(str) + "|"
           + df["frame"].astype(str)).to_numpy()
    o = np.argsort(key, kind="stable"); _, st = np.unique(key[o], return_index=True)
    li, ri = [], []
    for g in np.split(o, st[1:]):
        l, r = g[eye[g] == "left"], g[eye[g] == "right"]
        if len(l) and len(r):
            li.append(l[0]); ri.append(r[0])
    li, ri = np.array(li), np.array(ri)

    # 쌍마다: 두 눈 결합 예측, 그 오차, 그리고 라벨 없는 불일치 크기
    comb = (pred[li] + pred[ri] * FLIP) / 2
    err = angular_error(comb, gaze[li])
    disagree = np.linalg.norm(DEG(pred[li] - pred[ri] * FLIP), axis=1)

    r_p, p_p = pearson(disagree, err)
    r_s, p_s = spearman(disagree, err)
    print(f"쌍 {len(li):,}개")
    print(f"  불일치 크기 ↔ 결합 예측의 오차: Pearson r = {r_p:+.3f} (p={p_p:.1e}), "
          f"Spearman ρ = {r_s:+.3f}")

    rng = np.random.default_rng(0)
    rows = []
    print(f"\n{'버림 %':>8}{'불일치 기준':>13}{'무작위':>10}{'오라클':>10}")
    for pct in DROP:
        if pct == 0:
            keep_d = keep_r = keep_o = np.ones(len(err), bool)
        else:
            k = int(len(err) * (1 - pct / 100))
            keep_d = np.zeros(len(err), bool); keep_d[np.argsort(disagree)[:k]] = True
            keep_r = np.zeros(len(err), bool); keep_r[rng.permutation(len(err))[:k]] = True
            keep_o = np.zeros(len(err), bool); keep_o[np.argsort(err)[:k]] = True
        print(f"{pct:>8}{err[keep_d].mean():>13.3f}{err[keep_r].mean():>10.3f}"
              f"{err[keep_o].mean():>10.3f}")
        rows.append([pct, round(float(err[keep_d].mean()), 4),
                     round(float(err[keep_r].mean()), 4),
                     round(float(err[keep_o].mean()), 4)])

    save_table(f"a16_abstain_{tag}",
               ["drop_pct", "err_by_disagreement", "err_random", "err_oracle"], rows)
    put_numbers(**{
        f"{tag}_a16_disagree_err_pearson": round(r_p, 4),
        f"{tag}_a16_disagree_err_spearman": round(r_s, 4),
        f"{tag}_a16_err_drop20_signal": rows[3][1],
        f"{tag}_a16_err_drop20_random": rows[3][2],
        f"{tag}_a16_err_drop20_oracle": rows[3][3],
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axes[0].hexbin(disagree, err, gridsize=45, cmap="Greys", bins="log", mincnt=1)
    axes[0].set(xlabel="좌·우눈 예측 불일치 크기 (°) — 라벨 불필요",
                ylabel="결합 예측의 각도 오차 (°)")
    axes[0].text(0.03, 0.95, f"ρ = {r_s:+.3f}", transform=axes[0].transAxes,
                 fontsize=7, va="top")
    axes[1].plot([r[0] for r in rows], [r[1] for r in rows], "o-",
                 c="#1f4e79", label="불일치 큰 것부터 버림")
    axes[1].plot([r[0] for r in rows], [r[2] for r in rows], "s--",
                 c="#8aa0b8", label="무작위로 버림")
    axes[1].plot([r[0] for r in rows], [r[3] for r in rows], "^:",
                 c="#c0504d", label="오라클 (라벨 필요)")
    axes[1].set(xlabel="버린 비율 (%)", ylabel="남은 샘플의 평균 오차 (°)")
    axes[1].legend(fontsize=6)
    save_fig(fig, "diag_29_abstain.png",
             {"drop_pct": [r[0] for r in rows], "signal": [r[1] for r in rows],
              "random": [r[2] for r in rows], "oracle": [r[3] for r in rows]})

    gain = rows[3][2] - rows[3][1]
    print(f"\n결론 (음성): 불일치 크기는 샘플별 오차를 **예측하지 못한다** "
          f"(ρ={r_s:+.3f}). 상위 20%를 버려도 무작위와 차이가 {gain:+.3f}°뿐이고, "
          f"오라클은 {rows[3][2] - rows[3][3]:.2f}°를 줄인다.")
    print("  이유: 불일치는 두 눈의 치우침 **차이**를 재는데, 결합 예측의 오차는 두 눈이")
    print("  **공유하는** 성분이 지배한다(A13: pitch 분산의 79%가 공유). 불일치는 그 성분에")
    print("  대해 구조적으로 눈이 멀어 있다. 사람 수준 offset(A10)은 알아내지만")
    print("  샘플 수준 신뢰도는 못 준다 — 같은 신호가 한쪽에는 되고 한쪽에는 안 된다.")


if __name__ == "__main__":
    run()
