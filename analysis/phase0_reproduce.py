"""Phase 0 — 사전 발견 4개를 재현해 수치를 확정한다. CPU 전용.

출력은 report_assets/numbers.json 의 일부가 된다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import PERSONS
from metrics import angular_error
from analysis.common import sig

OUT = {}


def load(run: str):
    with np.load(f"results/{run}/predictions.npz") as z:
        person, pred, gaze, head = z["person"], z["pred"], z["gaze"], z["head"]
    with open(f"results/{run}/lopo.json", encoding="utf-8") as f:
        meta = json.load(f)
    return person, pred, gaze, head, meta


def f1_person_offset(run: str, tag: str) -> None:
    """발견 1 — 사람별 고정 offset(kappa angle)."""
    person, pred, gaze, _, _ = load(run)
    err = angular_error(pred, gaze)
    resid = np.degrees(pred - gaze)

    print(f"\n[{tag}] 발견 1 — 사람별 고정 offset")
    print(f"  전체 잔차 평균: yaw {resid[:, 0].mean():+.3f}°  pitch {resid[:, 1].mean():+.3f}°")

    rows = []
    for p in PERSONS:
        m = person == p
        rows.append((p, resid[m, 0].mean(), resid[m, 1].mean(),
                     np.hypot(resid[m, 0].mean(), resid[m, 1].mean()), err[m].mean()))

    print(f"  {'person':>7}{'yaw off':>10}{'pitch off':>11}{'|off|':>8}{'fold 오차':>10}")
    for p, dy, dp, mag, e in rows:
        print(f"  {p:>7}{dy:>+10.2f}{dp:>+11.2f}{mag:>8.2f}{e:>10.2f}")

    # oracle: 사람별 평균 잔차를 빼면 (테스트 라벨을 쓰므로 [진단용])
    corrected = pred.copy()
    for p in PERSONS:
        m = person == p
        corrected[m] -= np.radians([resid[m, 0].mean(), resid[m, 1].mean()])
    oracle = angular_error(corrected, gaze)

    mags = np.array([r[3] for r in rows])
    fold_err = np.array([r[4] for r in rows])
    r = np.corrcoef(mags, fold_err)[0, 1]
    # 피어슨 상관의 양측 p값 (t = r*sqrt(n-2)/sqrt(1-r^2), df=n-2)
    from scipy import stats
    t = r * np.sqrt(len(mags) - 2) / np.sqrt(1 - r ** 2)
    p_val = 2 * stats.t.sf(abs(t), len(mags) - 2)

    print(f"\n  원래 오차        {err.mean():.3f}°")
    print(f"  offset 제거 후   {oracle.mean():.3f}°   (회수 {err.mean() - oracle.mean():.3f}°)")
    print(f"  |offset| ~ fold 오차 상관 r = {r:+.3f}  (p = {p_val:.4f})")
    worst = sorted(rows, key=lambda t_: -t_[4])[:2]
    print(f"  최악 fold 2개: " + ", ".join(f"{w[0]} 오차 {w[4]:.2f}° |off| {w[3]:.2f}°" for w in worst))

    OUT[f"{tag}_error"] = round(float(err.mean()), 4)
    OUT[f"{tag}_offset_oracle_error"] = round(float(oracle.mean()), 4)
    OUT[f"{tag}_offset_recovery_deg"] = round(float(err.mean() - oracle.mean()), 4)
    OUT[f"{tag}_offset_mag_vs_folderr_r"] = round(float(r), 4)
    OUT[f"{tag}_offset_mag_vs_folderr_p"] = sig(p_val)
    OUT[f"{tag}_person_offsets_deg"] = {p: [round(float(dy), 4), round(float(dp), 4)]
                                        for p, dy, dp, _, _ in rows}


def f2_regression_dilution(run: str, tag: str) -> None:
    """발견 2 — '수축'의 상당 부분은 regression dilution."""
    person, pred, gaze, _, _ = load(run)
    gd, pd_ = np.degrees(gaze), np.degrees(pred)

    print(f"\n[{tag}] 발견 2 — regression dilution")
    for k, name in enumerate(("yaw", "pitch")):
        fwd = np.polyfit(gd[:, k], pd_[:, k], 1)[0]      # pred ~ gt
        rev = np.polyfit(pd_[:, k], gd[:, k], 1)[0]      # gt ~ pred
        per_fold = [np.polyfit(pd_[person == p, k], gd[person == p, k], 1)[0] for p in PERSONS]
        per_fold = np.array(per_fold)
        print(f"  {name:<6} pred~gt {fwd:.3f}   gt~pred(전체) {rev:.3f}   "
              f"gt~pred(fold) 중앙값 {np.median(per_fold):.3f} "
              f"[{per_fold.min():.2f}, {per_fold.max():.2f}]")
        OUT[f"{tag}_slope_pred_on_gt_{name}"] = round(float(fwd), 4)
        OUT[f"{tag}_slope_gt_on_pred_{name}"] = round(float(rev), 4)
        OUT[f"{tag}_slope_gt_on_pred_fold_median_{name}"] = round(float(np.median(per_fold)), 4)
        n_above = int((per_fold >= 1.0).sum())
        OUT[f"{tag}_folds_with_gt_on_pred_slope_ge1_{name}"] = n_above
        print(f"         fold 15개 중 {n_above}개가 1.0 이상")


def f3_epoch_selection_noise(run: str, tag: str) -> None:
    """발견 3 — val 기반 epoch 선택은 잡음이다."""
    _, _, _, _, meta = load(run)
    res = sorted(meta["results"], key=lambda r: r["test_person"])
    hist = np.array([[h["test"] for h in r["history"]] for r in res])   # (15, E)
    chosen = np.array([r["test_error"] for r in res])
    last = np.array([r["history"][-1]["test"] for r in res])
    best_ep = [r["best_epoch"] for r in res]

    print(f"\n[{tag}] 발견 3 — epoch 선택")
    print(f"  epoch별 15-fold 평균 test: " + " ".join(f"{v:.2f}" for v in hist.mean(0)))
    print(f"  val로 고른 epoch {best_ep}")
    print(f"  val 선택 {chosen.mean():.3f}°   마지막 epoch {last.mean():.3f}°   "
          f"차이 {chosen.mean() - last.mean():+.3f}°")
    print(f"  oracle(각 fold 최선 epoch) {hist.min(1).mean():.3f}°")

    OUT[f"{tag}_test_by_epoch"] = [round(float(v), 4) for v in hist.mean(0)]
    OUT[f"{tag}_val_selected_error"] = round(float(chosen.mean()), 4)
    OUT[f"{tag}_last_epoch_error"] = round(float(last.mean()), 4)
    OUT[f"{tag}_val_minus_last"] = round(float(chosen.mean() - last.mean()), 4)
    OUT[f"{tag}_oracle_epoch_error"] = round(float(hist.min(1).mean()), 4)
    OUT[f"{tag}_best_epochs"] = best_ep


def f4_throughput() -> None:
    """발견 4 — per-gpu 패킹 이득."""
    print("\n[공통] 발견 4 — per-gpu 패킹")
    import re
    info = {}
    for job, label in ((1330, "per-gpu 1 (job 1330, 중단됨)"), (1336, "per-gpu 2 (job 1336)")):
        path = f"slurm/logs/gaze-train-{job}.out"
        if not os.path.exists(path):
            print(f"  {label}: 로그 없음")
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        mins = [float(m) for m in re.findall(r"\((\d+\.\d+)분\)", text)]
        info[job] = (label, len(mins), max(mins) if mins else None)
        print(f"  {label}: epoch 기록 {len(mins)}개, 최대 누적 {max(mins) if mins else '-':.1f}분")
    OUT["throughput_note"] = ("job 1330은 중단되어 완주 기록이 없다. "
                              "Phase 1 B0에서 직접 측정한다.")


if __name__ == "__main__":
    for run, tag in (("base", "runA"), ("final", "runB")):
        f1_person_offset(run, tag)
        f2_regression_dilution(run, tag)
        f3_epoch_selection_noise(run, tag)
    f4_throughput()

    os.makedirs("report_assets", exist_ok=True)
    path = "report_assets/numbers.json"
    existing = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    existing.update(OUT)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"\n→ {path} ({len(OUT)}개 키)")
