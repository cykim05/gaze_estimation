"""분석용 데이터 팩 — report_assets/data/ 를 만든다.

보고서를 쓰는 사람이 원하는 각도로 다시 자를 수 있도록, 집계된 표가 아니라
**long format 원자료**를 낸다. 네 개 층으로 나눈다.

  runs.csv        실행 × fold          21 실행 × 15 fold = 315 행
  epochs.csv      실행 × fold × epoch  315 × 20 = 6,300 행
  samples.csv.gz  최종 실행의 샘플별   45,000 행 (잔차·조건·메타)
  persons.csv     사람별 종합          15 행 (offset·난이도·조건 요약)

runs/epochs는 전부 `lopo.json`에서, samples는 `predictions.npz` + 라벨에서 온다.
집계는 하지 않는다 — 평균·검정은 쓰는 쪽에서 한다.
"""
from __future__ import annotations

import glob
import gzip
import json
import os

import numpy as np

from analysis.common import (DEG, PERSONS, ROOT, angular_error, load_run,
                             put_numbers, section)

OUT = os.path.join(ROOT, "report_assets", "data")

# 실행 이름 → (설계 묶음, seed). 묶음이 같으면 '같은 설정의 반복'이다.
GROUPS = {
    "p1_seed0": ("base", 0), "p1_seed1": ("base", 1), "p1_seed2": ("base", 2),
    "p4_seed3": ("base", 3), "p4_seed4": ("base", 4), "p4_seed5": ("base", 5),
    "p4_seed6": ("base", 6),
    "final":    ("base", 0),          # = p1_seed0 과 비트 단위 동일
    "base":     ("prev_A", 0),        # 이전 설정 (b128/do0.2/15ep)
    "ab_pool2": ("ab_pool2", 0), "ab_pool2_s1": ("ab_pool2", 1),
    "ab_pool2_s2": ("ab_pool2", 2),
    "ab_pixnorm": ("ab_pixnorm", 0), "ab_pixnorm_s1": ("ab_pixnorm", 1),
    "ab_nohead": ("ab_nohead", 0), "ab_nohead_s1": ("ab_nohead", 1),
    "ab_angular": ("ab_angular", 0), "ab_l1": ("ab_l1", 0),
    "ab_k5": ("ab_k5", 0), "ab_narrow": ("ab_narrow", 0),
    "ab_dropout02": ("ab_dropout02", 0),
}
CFG_COLS = ["epochs", "batch_size", "lr", "dropout", "weight_decay", "loss",
            "kernel", "n_pools", "use_head", "pixel_norm", "conv_channels",
            "hidden", "augment", "n_val_persons", "seed"]


def _write(name: str, header: list[str], rows: list[list], gz=False) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    op = gzip.open if gz else open
    with op(path, "wt", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print(f"  {name:<18} {len(rows):>6}행 × {len(header):>2}열  "
          f"({os.path.getsize(path) / 1024:.0f} KB)")
    return path


def runs_and_epochs() -> tuple[int, int]:
    """실행 × fold, 실행 × fold × epoch."""
    dirs = sorted(d for d in glob.glob("results/*")
                  if os.path.exists(f"{d}/lopo.json")
                  and os.path.basename(d) in GROUPS)
    run_rows, ep_rows = [], []
    for d in dirs:
        name = os.path.basename(d)
        group, seed = GROUPS[name]
        meta = json.load(open(f"{d}/lopo.json", encoding="utf-8"))
        cfg = meta["config"]
        for r in meta["results"]:
            run_rows.append([
                name, group, seed, r["test_person"],
                "|".join(r["val_persons"]),
                round(r["test_error"], 6), round(r["val_error"], 6),
                r["best_epoch"], round(r["final_test_error"], 6),
                round(r.get("seconds", float("nan")), 1),
            ] + [cfg.get(c) for c in CFG_COLS])
            for h in r["history"]:
                ep_rows.append([name, group, seed, r["test_person"], h["epoch"],
                                round(h["loss"], 8), round(h["val"], 6),
                                round(h["test"], 6)])
    _write("runs.csv",
           ["run", "config_group", "seed", "test_person", "val_persons",
            "test_error_valsel", "val_error", "best_epoch",
            "test_error_lastep", "seconds"] + CFG_COLS, run_rows)
    _write("epochs.csv",
           ["run", "config_group", "seed", "test_person", "epoch",
            "train_loss", "val_error", "test_error"], ep_rows)
    return len(run_rows), len(ep_rows)


def samples(run_dir="results/final") -> int:
    """최종 실행의 샘플별 잔차 + 조건·메타 (라벨 파일에서 합친다)."""
    import pandas as pd

    import dataset
    d = load_run(run_dir)
    df = dataset.label_frame()
    # predictions.npz 의 순서는 사람별 라벨 순서와 같다 (dataset 로더 순서).
    assert len(df) == len(d["person"]), f"{len(df)} vs {len(d['person'])}"
    assert (df["person"].to_numpy() == d["person"]).all(), "사람 순서 불일치"

    pr, gz = DEG(d["pred"]), DEG(d["gaze"])
    out = pd.DataFrame({
        "person": d["person"],
        "which_eye": df["which_eye"].to_numpy(),
        "day": df["day"].to_numpy(),
        "frame": df["frame"].to_numpy(),
        "gaze_yaw_deg": gz[:, 0], "gaze_pitch_deg": gz[:, 1],
        "pred_yaw_deg": pr[:, 0], "pred_pitch_deg": pr[:, 1],
        "error_deg": d["err"],
        "head_yaw_deg": DEG(df["2DHead_yaw"].to_numpy()),
        "head_pitch_deg": DEG(df["2DHead_pitch"].to_numpy()),
        "eye_cam_dist_mm": df["GazeOrigin_z"].to_numpy(),
    }).round(5)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "samples_final.csv.gz")
    out.to_csv(path, index=False, compression="gzip")
    print(f"  {'samples_final.csv.gz':<18} {len(out):>6}행 × {out.shape[1]:>2}열  "
          f"({os.path.getsize(path) / 1024:.0f} KB)")
    return len(out)


def persons() -> int:
    """사람 단위 종합 — 기존 그림 CSV들을 사람 키로 합친다."""
    import csv

    def read(name, key="person"):
        path = os.path.join(ROOT, "report_assets", name)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return {r[key]: r for r in csv.DictReader(f) if r.get(key)}

    off = read("figures/12_person_offset.csv")
    free = read("figures/diag_22_labelfree.csv")

    import dataset
    df = dataset.label_frame()
    runs = {}
    for name in ("p1_seed0", "p1_seed1", "p1_seed2",
                 "p4_seed3", "p4_seed4", "p4_seed5", "p4_seed6"):
        meta = json.load(open(f"results/{name}/lopo.json", encoding="utf-8"))
        runs[name] = {r["test_person"]: r["test_error"] for r in meta["results"]}

    rows = []
    for p in PERSONS:
        sub = df[df["person"] == p]
        errs = [runs[n][p] for n in runs]
        o = off.get(p, {})
        rows.append([
            p, len(sub), sub["day"].nunique(),
            round(float(DEG(sub["2DGaze_yaw"]).mean()), 5),
            round(float(DEG(sub["2DGaze_pitch"]).mean()), 5),
            round(float(DEG(sub["2DGaze_yaw"]).std(ddof=1)), 5),
            round(float(DEG(sub["2DGaze_pitch"]).std(ddof=1)), 5),
            round(float(DEG(sub["2DHead_yaw"]).mean()), 5),
            round(float(DEG(sub["2DHead_pitch"]).mean()), 5),
            round(float(np.mean(errs)), 5), round(float(np.std(errs, ddof=1)), 5),
            round(float(min(errs)), 5), round(float(max(errs)), 5),
            o.get("yaw_off"), o.get("pitch_off"), o.get("offset_mag"),
            o.get("fold_err_offset_removed"),
            (free.get(p) or {}).get("disagree_yaw"),
            (free.get(p) or {}).get("disagree_pitch"),
        ])
    _write("persons.csv",
           ["person", "n_samples", "n_days",
            "gaze_yaw_mean_deg", "gaze_pitch_mean_deg",
            "gaze_yaw_sd_deg", "gaze_pitch_sd_deg",
            "head_yaw_mean_deg", "head_pitch_mean_deg",
            "fold_error_mean_7seeds", "fold_error_sd_7seeds",
            "fold_error_min", "fold_error_max",
            "offset_yaw_deg", "offset_pitch_deg", "offset_mag_deg",
            "fold_error_offset_removed", "lr_disagree_yaw_deg",
            "lr_disagree_pitch_deg"], rows)
    return len(rows)


def run():
    section("데이터 팩 — report_assets/data/")
    n_run, n_ep = runs_and_epochs()
    n_s = samples()
    n_p = persons()
    print(f"\n합계 {n_run + n_ep + n_s + n_p:,}행. 집계하지 않은 long format이다.")
    put_numbers(pack_rows_runs=n_run, pack_rows_epochs=n_ep,
                pack_rows_samples=n_s, pack_rows_persons=n_p)


if __name__ == "__main__":
    run()
