"""A10 — 라벨 없는 신호로 사람 offset·난이도를 알 수 있나? (자율 탐색)

동기: A1에서 사람별 offset이 오차의 17%를 차지하고 fold 난이도를 예측했다(r=+0.67).
A2는 라벨 10장이면 75%를 회수한다고 했다. 그런데 라벨은 못 쓰는 것이 원칙이다.
**라벨 없이 얻을 수 있는 신호**로 offset을 추정할 수 있다면 이야기가 달라진다.

후보 신호 (전부 테스트 사람의 **라벨을 쓰지 않는다**):
  S1 좌우 눈 불일치  — 같은 프레임의 두 예측이 정렬 후 얼마나 어긋나나 [진단용: 프레임 묶음]
  S2 예측 분산      — 그 사람 예측의 퍼짐
  S3 예측 평균      — 그 사람 예측의 중심 (학습 분포 중심에서 얼마나 떨어졌나)
  S4 특징 거리      — penultimate 특징이 학습 사람들 평균에서 얼마나 떨어졌나

각 신호와 (a) 실제 offset, (b) fold 오차의 상관을 15명에서 잰다.
"""
from __future__ import annotations

import os

import numpy as np

from analysis.common import (DEG, PERSONS, angular_error, load_folds, load_run,
                             pearson, plt, put_numbers, save_fig, save_table, section)

FLIP = np.array([-1.0, 1.0], dtype=np.float32)


def run(run_dir="results/final", tag="runB"):
    section(f"A10 — 라벨 없는 offset·난이도 신호 ({run_dir})")
    import dataset
    d = load_run(run_dir)
    df = dataset.label_frame()
    person, pred, gaze, err = d["person"], d["pred"], d["gaze"], d["err"]

    # 정답 (비교 대상)
    off = np.array([(pred[person == p] - gaze[person == p]).mean(0) for p in PERSONS])
    off_mag = np.hypot(DEG(off[:, 0]), DEG(off[:, 1]))
    fold_err = d["fold_err"]

    eye = df["which_eye"].to_numpy()
    key = (df["person"].astype(str) + "|" + df["day"].astype(str) + "|"
           + df["frame"].astype(str)).to_numpy()
    order = np.argsort(key, kind="stable")
    _, starts = np.unique(key[order], return_index=True)
    li, ri = [], []
    for g in np.split(order, starts[1:]):
        l, r = g[eye[g] == "left"], g[eye[g] == "right"]
        if len(l) and len(r):
            li.append(l[0]); ri.append(r[0])
    li, ri = np.array(li), np.array(ri)
    disagree = DEG(pred[li] - pred[ri] * FLIP)        # (쌍, 2) 라벨을 안 쓴다

    sig = {}
    sig["S1 좌우 불일치 |평균|"] = np.array([
        np.hypot(*disagree[person[li] == p].mean(0)) for p in PERSONS])
    sig["S1b 좌우 불일치 yaw 평균"] = np.array([
        disagree[person[li] == p][:, 0].mean() for p in PERSONS])
    sig["S1c 좌우 불일치 pitch 평균"] = np.array([
        disagree[person[li] == p][:, 1].mean() for p in PERSONS])
    sig["S2 예측 표준편차"] = np.array([
        DEG(pred[person == p]).std(0).mean() for p in PERSONS])
    sig["S3 예측 평균 이탈"] = np.array([
        np.hypot(*(DEG(pred[person == p]).mean(0) - DEG(pred).mean(0))) for p in PERSONS])

    # S4 특징 거리 (특징이 저장된 실행에서만)
    feats = load_folds(run_dir, keys=["feat_test", "feat_train"])
    if feats and all("feat_test" in feats.get(p, {}) for p in PERSONS):
        s4 = []
        for p in PERSONS:
            ft, fr = feats[p]["feat_test"], feats[p]["feat_train"]
            s4.append(float(np.linalg.norm(ft.mean(0) - fr.mean(0))))
        sig["S4 특징 거리"] = np.array(s4)
    else:
        print("  (S4: 특징이 저장되지 않은 실행이라 건너뛴다)")

    print(f"\n{'신호':<26}{'↔ |offset|':>14}{'↔ fold 오차':>15}")
    rows = []
    for name, v in sig.items():
        r1, p1 = pearson(np.abs(v) if "yaw" in name or "pitch" in name else v, off_mag)
        r2, p2 = pearson(np.abs(v) if "yaw" in name or "pitch" in name else v, fold_err)
        star1 = "*" if p1 < 0.05 else " "
        star2 = "*" if p2 < 0.05 else " "
        print(f"{name:<26}  r={r1:+.3f}{star1} (p={p1:.3f}) r={r2:+.3f}{star2} (p={p2:.3f})")
        rows.append([name, round(r1, 4), round(p1, 5), round(r2, 4), round(p2, 5)])

    # 부호까지 맞추는지 — yaw/pitch 성분별로 직접 비교
    print(f"\n성분별 (부호 포함): 좌우 불일치 ↔ 실제 offset")
    comp = []
    for k, axis in enumerate(("yaw", "pitch")):
        s = np.array([disagree[person[li] == p][:, k].mean() for p in PERSONS])
        r, pv = pearson(s, DEG(off[:, k]))
        print(f"  {axis:<6} r = {r:+.3f}  p = {pv:.4f}  "
              f"{'← 라벨 없이 offset 방향을 안다' if pv < 0.05 else ''}")
        comp.append([axis, round(r, 4), round(pv, 5)])
        put_numbers(**{f"{tag}_a10_disagree_vs_offset_{axis}_r": round(r, 4),
                       f"{tag}_a10_disagree_vs_offset_{axis}_p": round(pv, 5)})

    save_table(f"a10_labelfree_signals_{tag}",
               ["signal", "r_vs_offset_mag", "p_vs_offset_mag",
                "r_vs_fold_err", "p_vs_fold_err"], rows)
    save_table(f"a10_disagree_component_{tag}",
               ["axis", "pearson_r_vs_offset", "p"], comp)

    best = max(rows, key=lambda r: abs(r[3]))
    put_numbers(**{f"{tag}_a10_best_signal": best[0],
                   f"{tag}_a10_best_signal_r_folderr": best[3],
                   f"{tag}_a10_best_signal_p_folderr": best[4]})

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, k, axis in zip(axes[:2], (0, 1), ("yaw", "pitch")):
        s = np.array([disagree[person[li] == p][:, k].mean() for p in PERSONS])
        ax.scatter(s, DEG(off[:, k]), s=30, c="#1f4e79")
        for i, p in enumerate(PERSONS):
            ax.annotate(p, (s[i], DEG(off[i, k])), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
        b = np.polyfit(s, DEG(off[:, k]), 1)
        xs = np.linspace(s.min(), s.max(), 10)
        ax.plot(xs, np.polyval(b, xs), "--", lw=0.9, c="#1f4e79")
        r, pv = pearson(s, DEG(off[:, k]))
        ax.set(xlabel=f"좌우 예측 불일치 {axis} 평균 (°) — 라벨 불필요",
               ylabel=f"실제 사람 offset {axis} (°)")
        ax.text(0.03, 0.95, f"r = {r:+.3f}, p = {pv:.3f}",
                transform=ax.transAxes, fontsize=7, va="top")
    ax = axes[2]
    names = [r[0] for r in rows]
    ax.barh(np.arange(len(rows)), [abs(r[3]) for r in rows], color="#1f4e79", height=0.6)
    ax.set_yticks(np.arange(len(rows)), names, fontsize=6)
    ax.set(xlabel="|상관계수| (fold 오차와)")
    ax.invert_yaxis()
    save_fig(fig, "diag_22_labelfree.png",
             {"person": PERSONS, "offset_yaw": DEG(off[:, 0]), "offset_pitch": DEG(off[:, 1]),
              "disagree_yaw": [disagree[person[li] == p][:, 0].mean() for p in PERSONS],
              "disagree_pitch": [disagree[person[li] == p][:, 1].mean() for p in PERSONS],
              "fold_err": fold_err})


if __name__ == "__main__":
    run()
