"""A17 — 사람 offset이 눈의 '생김새'로 설명되나. **[진단용]** (자율 탐색)

A12는 pitch offset이 사람 수준 상수(kappa angle 같은 해부학적 양)처럼 행동한다고 했다.
해부학적이라면 **눈 이미지에 흔적이 남아야 한다**. 사람별 평균 눈 이미지를 주성분으로
줄이고, 그 성분이 offset을 예측하는지 본다.

표본이 15명뿐이라 과적합이 쉽다. leave-one-person-out으로 예측해 정직하게 잰다.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (sig, DEG, PERSONS, load_run, pearson, plt, put_numbers,
                             save_fig, save_table, section)

N_PC = 4


def run(run_dir="results/final", tag="runB"):
    section(f"A17 — offset이 눈 생김새로 설명되나 [진단용] ({run_dir})")
    import dataset
    d = load_run(run_dir)
    person, pred, gaze = d["person"], d["pred"], d["gaze"]
    off = np.array([DEG(pred[person == p] - gaze[person == p]).mean(0) for p in PERSONS])

    samples = dataset.load(verbose=False)
    mean_img = np.stack([samples.only(p).image[:, 0].mean(0).ravel() / 255.0
                         for p in PERSONS])              # (15, 2160)
    X = mean_img - mean_img.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :N_PC] * S[:N_PC]                       # (15, N_PC)
    var = (S ** 2 / (S ** 2).sum())[:N_PC]
    print(f"사람별 평균 눈 이미지의 주성분 (15명)")
    for k in range(N_PC):
        print(f"  PC{k+1} 분산 {var[k]:.1%}", end="")
        for j, axis in enumerate(("yaw", "pitch")):
            r, pv = pearson(scores[:, k], off[:, j])
            print(f"   ↔ offset {axis} r={r:+.3f}{'*' if pv < 0.05 else ' '}(p={pv:.3f})",
                  end="")
        print()

    rows = []
    for k in range(N_PC):
        row = [f"PC{k+1}", round(float(var[k]), 4)]
        for j in range(2):
            r, pv = pearson(scores[:, k], off[:, j])
            row += [round(r, 4), sig(pv)]
        rows.append(row)
    save_table(f"a17_appearance_pcs_{tag}",
               ["pc", "var_frac", "r_offset_yaw", "p_yaw", "r_offset_pitch", "p_pitch"],
               rows)

    # 정직한 예측: leave-one-person-out 선형 회귀
    print(f"\nleave-one-person-out 예측 (PC {N_PC}개 → offset)")
    for j, axis in enumerate(("yaw", "pitch")):
        pred_off = np.zeros(len(PERSONS))
        for i in range(len(PERSONS)):
            tr = np.setdiff1d(np.arange(len(PERSONS)), [i])
            A = np.c_[scores[tr], np.ones(len(tr))]
            w = np.linalg.lstsq(A, off[tr, j], rcond=None)[0]
            pred_off[i] = np.r_[scores[i], 1.0] @ w
        r, pv = pearson(pred_off, off[:, j])
        rmse = float(np.sqrt(((pred_off - off[:, j]) ** 2).mean()))
        base_rmse = float(np.sqrt(((off[:, j] - off[:, j].mean()) ** 2).mean()))
        print(f"  {axis:<6} 예측 ↔ 실제 r = {r:+.3f} (p={pv:.3f})   "
              f"RMSE {rmse:.2f}° (상수 예측 {base_rmse:.2f}°)")
        put_numbers(**{f"{tag}_a17_loo_{axis}_r": round(r, 4),
                       f"{tag}_a17_loo_{axis}_p": sig(pv),
                       f"{tag}_a17_loo_{axis}_rmse": round(rmse, 4),
                       f"{tag}_a17_loo_{axis}_rmse_base": round(base_rmse, 4)})

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for k in range(2):
        axes[k].imshow(Vt[k].reshape(36, 60), cmap="RdBu_r")
        rr, pp = pearson(scores[:, k], off[:, 1])
        axes[k].set_title(f"PC{k+1} ({var[k]:.0%}) — offset pitch와 r={rr:+.2f}",
                          fontsize=7)
        axes[k].set_xticks([]); axes[k].set_yticks([]); axes[k].grid(False)
    best = int(np.argmax([abs(pearson(scores[:, k], off[:, 1])[0]) for k in range(N_PC)]))
    axes[2].scatter(scores[:, best], off[:, 1], s=30, c="#1f4e79")
    for i, p in enumerate(PERSONS):
        axes[2].annotate(p, (scores[i, best], off[i, 1]), fontsize=6,
                         xytext=(3, 2), textcoords="offset points")
    rr, pp = pearson(scores[:, best], off[:, 1])
    xs = np.linspace(scores[:, best].min(), scores[:, best].max(), 10)
    axes[2].plot(xs, np.polyval(np.polyfit(scores[:, best], off[:, 1], 1), xs),
                 "--", lw=0.9, c="#1f4e79")
    axes[2].set(xlabel=f"사람 평균 눈 이미지의 PC{best+1} 점수",
                ylabel="사람 offset pitch (°)")
    axes[2].text(0.03, 0.95, f"r = {rr:+.3f}, p = {pp:.3f}",
                 transform=axes[2].transAxes, fontsize=7, va="top")
    save_fig(fig, "diag_30_appearance_offset.png",
             {"person": PERSONS, "pc_score": scores[:, best],
              "offset_pitch": off[:, 1]})


if __name__ == "__main__":
    run()
