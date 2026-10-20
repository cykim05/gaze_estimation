"""결과 파일 정합성 검사 — 새 결과가 나올 때마다 반드시 통과해야 한다.

Phase 0에서 results/base의 predictions.npz가 제 lopo.json과 15 fold 중 11개에서
어긋나 있는 것을 발견했다. 분석은 전부 predictions.npz를 읽으므로, 이 검사를
통과하지 못한 실행은 분석에 쓰지 않는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import PERSONS
from metrics import angular_error

TOL = 0.005   # 도. 반올림 오차보다 크면 불일치로 본다


def check(run_dir: str, verbose=True) -> tuple[bool, list]:
    npz = os.path.join(run_dir, "predictions.npz")
    js = os.path.join(run_dir, "lopo.json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        return False, [("파일 없음", run_dir, 0.0)]

    with np.load(npz) as z:
        person, pred, gaze = z["person"], z["pred"], z["gaze"]
    meta = json.load(open(js, encoding="utf-8"))
    reported = {r["test_person"]: r["test_error"] for r in meta["results"]}

    bad = []
    for p in PERSONS:
        if p not in reported:
            continue
        m = person == p
        if not m.any():
            bad.append((p, float("nan"), float("nan")))
            continue
        recomputed = float(angular_error(pred[m], gaze[m]).mean())
        diff = recomputed - reported[p]
        if abs(diff) > TOL:
            bad.append((p, recomputed, reported[p]))

    ok = not bad
    if verbose:
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {run_dir}: {len(reported)} fold 중 불일치 {len(bad)}개")
        for p, rc, rp in bad:
            print(f"         {p}  npz {rc:.3f}°  json {rp:.3f}°  차이 {rc - rp:+.3f}°")
    return ok, bad


if __name__ == "__main__":
    targets = sys.argv[1:] or [os.path.join("results", d) for d in sorted(os.listdir("results"))
                               if os.path.exists(os.path.join("results", d, "lopo.json"))]
    all_ok = True
    for t in targets:
        ok, _ = check(t)
        all_ok &= ok
    print("\n전부 일치" if all_ok else "\n불일치 있음 — 해당 실행은 분석에 쓰지 않는다")
    sys.exit(0 if all_ok else 1)
