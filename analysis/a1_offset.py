"""A1 — 오차 분해: 사람 고정 offset 성분 vs 샘플 성분. [일부 진단용]

질문: LOPO 오차 중 얼마가 '그 사람에게 일정한 치우침'이고 얼마가 '샘플마다 다른 잡음'인가.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (sig, DEG, FIG, PERSONS, angular_error, load_run, pearson,
                             plt, put_numbers, save_fig, save_table, section)


def run(run_dir="results/final", tag="runB"):
    section(f"A1 — 오차 분해 ({run_dir})")
    d = load_run(run_dir)
    person, pred, gaze, err = d["person"], d["pred"], d["gaze"], d["err"]
    resid = DEG(pred - gaze)                       # (N, 2) 도

    off = np.array([resid[person == p].mean(0) for p in PERSONS])      # (15, 2)
    mag = np.hypot(off[:, 0], off[:, 1])

    # 분산 분해: 사람별 평균(between) vs 사람 안 편차(within)
    within = np.concatenate([resid[person == p] - off[i] for i, p in enumerate(PERSONS)])
    var_b = off.var(axis=0, ddof=0)
    var_w = within.var(axis=0, ddof=0)
    share = var_b / (var_b + var_w)

    print(f"잔차 분산 분해 (도²)")
    print(f"  {'축':<8}{'between(사람)':>14}{'within(샘플)':>14}{'between 비중':>14}")
    for k, name in enumerate(("yaw", "pitch")):
        print(f"  {name:<8}{var_b[k]:>14.3f}{var_w[k]:>14.3f}{share[k]:>13.1%}")

    # [진단용] oracle: 테스트 사람 라벨로 offset을 추정해 뺀다
    corrected = pred - np.radians(off[[PERSONS.index(p) for p in person]])
    oracle = angular_error(corrected, gaze)
    fold_err = np.array([err[person == p].mean() for p in PERSONS])
    fold_or = np.array([oracle[person == p].mean() for p in PERSONS])

    r, pv = pearson(mag, fold_err)
    print(f"\n원래 {err.mean():.3f}°  →  offset 제거 {oracle.mean():.3f}°  "
          f"(회수 {err.mean() - oracle.mean():.3f}°, {1 - oracle.mean()/err.mean():.1%})")
    print(f"|offset| ~ fold 오차 상관 r = {r:+.3f}  p = {pv:.4f}")

    put_numbers(**{
        f"{tag}_resid_var_between_yaw": round(float(var_b[0]), 4),
        f"{tag}_resid_var_within_yaw": round(float(var_w[0]), 4),
        f"{tag}_resid_var_between_pitch": round(float(var_b[1]), 4),
        f"{tag}_resid_var_within_pitch": round(float(var_w[1]), 4),
        f"{tag}_offset_share_yaw": round(float(share[0]), 4),
        f"{tag}_offset_share_pitch": round(float(share[1]), 4),
        f"{tag}_offset_oracle_error": round(float(oracle.mean()), 4),
        f"{tag}_offset_recovery_deg": round(float(err.mean() - oracle.mean()), 4),
        f"{tag}_offset_recovery_pct": round(float(1 - oracle.mean() / err.mean()), 4),
        f"{tag}_offset_mag_vs_folderr_r": round(r, 4),
        f"{tag}_offset_mag_vs_folderr_p": sig(pv),
        f"{tag}_offset_mag_max": round(float(mag.max()), 4),
        f"{tag}_offset_mag_max_person": PERSONS[int(mag.argmax())],
    })

    save_table(f"a1_person_offset_{tag}",
               ["person", "yaw_offset_deg", "pitch_offset_deg", "offset_mag_deg",
                "fold_error_deg", "fold_error_after_offset_deg"],
               [[p, round(float(off[i, 0]), 3), round(float(off[i, 1]), 3),
                 round(float(mag[i]), 3), round(float(fold_err[i]), 3),
                 round(float(fold_or[i]), 3)] for i, p in enumerate(PERSONS)])

    # 그림 1: offset 화살표
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax = axes[0]
    for i, p in enumerate(PERSONS):
        ax.annotate("", xy=(off[i, 0], off[i, 1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color="#1f4e79"))
        ax.annotate(p, (off[i, 0], off[i, 1]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    lim = mag.max() * 1.25
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
           xlabel="yaw 평균 잔차 (°)", ylabel="pitch 평균 잔차 (°)")
    ax.axhline(0, lw=0.5, c="k"); ax.axvline(0, lw=0.5, c="k")
    ax.set_aspect("equal")

    ax = axes[1]
    ax.scatter(mag, fold_err, s=30, c="#1f4e79", label="원래")
    ax.scatter(mag, fold_or, s=30, c="#c0504d", marker="^", label="offset 제거 [진단용]")
    for i, p in enumerate(PERSONS):
        ax.annotate(p, (mag[i], fold_err[i]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    b = np.polyfit(mag, fold_err, 1)
    xs = np.linspace(0, mag.max() * 1.05, 10)
    ax.plot(xs, np.polyval(b, xs), "--", lw=0.9, c="#1f4e79")
    ax.set(xlabel="사람 offset 크기 |offset| (°)", ylabel="fold 평균 각도 오차 (°)")
    ax.legend(fontsize=6)
    ax.text(0.03, 0.95, f"r = {r:+.3f}, p = {pv:.3f}", transform=ax.transAxes,
            fontsize=7, va="top")
    save_fig(fig, "12_person_offset.png",
             {"person": PERSONS, "yaw_off": off[:, 0], "pitch_off": off[:, 1],
              "offset_mag": mag, "fold_err": fold_err, "fold_err_offset_removed": fold_or})

    print(f"\n결론: 잔차 분산의 {share[0]:.0%}(yaw) / {share[1]:.0%}(pitch)가 사람 단위 고정 "
          f"성분이고, 그것만 빼도 오차가 {1 - oracle.mean()/err.mean():.0%} 줄어든다.")
    return {"off": off, "mag": mag, "fold_err": fold_err, "oracle": oracle}


if __name__ == "__main__":
    run()
