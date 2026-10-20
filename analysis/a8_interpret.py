"""A8 — 해석: 모델은 무엇을 보나. interpret.npz(GPU eval-only 산출물)를 읽는다."""
from __future__ import annotations

import glob
import os

import numpy as np

from analysis.common import (PERSONS, plt, put_numbers, save_fig, save_table, section)


def run(run_dir=None):
    section("A8 — 해석 (occlusion · head pose · person probe)")
    if run_dir is None:
        cands = sorted(glob.glob("results/*/interpret.npz"))
        if not cands:
            print("  interpret.npz가 아직 없다 (GPU eval-only job 필요).")
            return
        path = cands[0]
    else:
        path = os.path.join(run_dir, "interpret.npz")
    if not os.path.exists(path):
        print(f"  {path} 없음."); return
    print(f"  {path}")

    with np.load(path, allow_pickle=True) as z:
        occl = z["occlusion"]; patch = int(z["patch"]); stride = int(z["stride"])
        head_rows = z["head_rows"]; probe_rows = z["probe_rows"]

    # 1) occlusion
    valid = occl[~np.all(occl == 0, axis=(1, 2))]
    m = valid.mean(0)
    H, W = 36, 60
    ys = np.arange(0, H - patch + 1, stride) + patch / 2
    xs = np.arange(0, W - patch + 1, stride) + patch / 2
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    print(f"\nocclusion ({patch}×{patch} 조각, stride {stride}, 사람 {len(valid)}명 평균)")
    print(f"  최대 상승 {m.max():+.3f}° @ (행 {ys[iy]:.0f}, 열 {xs[ix]:.0f})")
    print(f"  최소 상승 {m.min():+.3f}° @ 가장자리")
    print(f"  가운데(동공) 근처 {m[len(ys)//2, len(xs)//2]:+.3f}°")

    # 2) head pose
    hr = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in head_rows])
    print(f"\nhead pose를 망가뜨렸을 때 오차 상승 (사람 {len(hr)}명)")
    print(f"  0으로 두기   {hr[:, 1].mean():+.3f}° ± {hr[:, 1].std(ddof=1)/np.sqrt(len(hr)):.3f}")
    print(f"  무작위 섞기  {hr[:, 2].mean():+.3f}° ± {hr[:, 2].std(ddof=1)/np.sqrt(len(hr)):.3f}")

    # 3) person probe
    if len(probe_rows):
        pr = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in probe_rows])
        print(f"\nperson-ID linear probe (penultimate 특징 → 학습 12명 분류)")
        print(f"  정확도 {pr[:, 1].mean():.3f} (무작위 {pr[:, 2].mean():.3f}, "
              f"{pr[:, 1].mean()/pr[:, 2].mean():.1f}배)")
        put_numbers(a8_person_probe_acc=round(float(pr[:, 1].mean()), 4),
                    a8_person_probe_chance=round(float(pr[:, 2].mean()), 4))
        save_table("a8_person_probe", ["person_fold", "n_classes", "accuracy", "chance"],
                   [[r[0], int(r[1]), round(float(r[2]), 4), round(float(r[3]), 4)]
                    for r in probe_rows])

    put_numbers(a8_occl_max_rise=round(float(m.max()), 4),
                a8_occl_max_row=float(ys[iy]), a8_occl_max_col=float(xs[ix]),
                a8_head_zero_rise=round(float(hr[:, 1].mean()), 4),
                a8_head_shuffle_rise=round(float(hr[:, 2].mean()), 4))
    save_table("a8_head_ablation",
               ["person_fold", "base_err_deg", "head_zero_rise_deg", "head_shuffle_rise_deg"],
               [[r[0], float(r[1]), float(r[2]), float(r[3])] for r in head_rows])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
    fig.subplots_adjust(wspace=0.45)
    im = axes[0].imshow(m, cmap="magma", extent=[0, W, H, 0], aspect="auto")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("가렸을 때 오차 상승 (°)", fontsize=7)
    # EDA가 픽셀 표준편차만으로 찍은 자리와, 정규화가 맞춰 둔 눈 중심
    axes[0].plot(19, 17, "o", mfc="none", mec="#4fd1c5", ms=11, mew=1.6)
    axes[0].annotate("EDA 예측 (17,19)", (19, 17), color="#4fd1c5", fontsize=6.5,
                     xytext=(8, -12), textcoords="offset points")
    axes[0].plot(30, 18, "x", color="#9ad0ff", ms=8, mew=1.6)
    axes[0].annotate("눈 중심(동공)", (30, 18), color="#9ad0ff", fontsize=6.5,
                     xytext=(6, 10), textcoords="offset points")
    axes[0].set(xlabel="이미지 열 (픽셀)", ylabel="이미지 행 (픽셀)")
    axes[0].grid(False)

    x = np.arange(len(hr)); w = 0.38
    axes[1].bar(x - w / 2, hr[:, 1], w, label="head pose = 0", color="#1f4e79")
    axes[1].bar(x + w / 2, hr[:, 2], w, label="head pose 섞기", color="#8aa0b8")
    axes[1].set_xticks(x, [r[0] for r in head_rows], rotation=90, fontsize=6)
    axes[1].axhline(0, lw=0.7, c="k")
    axes[1].set(ylabel="head pose를 망가뜨렸을 때\n오차 상승 (°)")
    axes[1].legend(fontsize=6)

    if len(probe_rows):
        axes[2].bar(np.arange(len(pr)), pr[:, 1], color="#1f4e79")
        axes[2].axhline(pr[0, 2], ls="--", lw=0.9, c="#c0504d", label="무작위")
        axes[2].set_xticks(np.arange(len(pr)), [r[0] for r in probe_rows],
                           rotation=90, fontsize=6)
        axes[2].set(ylabel="person-ID probe 정확도")
        axes[2].legend(fontsize=6)
    save_fig(fig, "24_interpretation.png",
             {"row": np.repeat(ys, len(xs)), "col": np.tile(xs, len(ys)),
              "occl_rise": m.ravel()})


if __name__ == "__main__":
    run()
