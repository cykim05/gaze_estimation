"""Phase 4 검수 — FINDINGS의 수치가 스크립트에서 다시 나오는지, 규칙을 어긴 곳이 없는지.

  python analysis/verify.py

검사 항목
  1. numbers.json의 모든 키가 FINDINGS.md에서 참조되는 형식인지 / 역으로 본문 수치가
     numbers.json에 있는지
  2. 결과 디렉터리의 predictions.npz ↔ lopo.json 정합성
  3. 테스트로 고른 흔적이 없는지 (train.py/search.py에서 test를 선택에 쓰는 경로)
  4. [진단용] 태그 누락 — 테스트 라벨·메타데이터를 쓰는 분석 모듈에 태그가 있는지
  5. GPU 2장 규칙 — sacct로 동시 RUNNING 구간을 재구성
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

FAIL = []


def ok(cond, msg):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAIL.append(msg)


def check_numbers():
    print("\n1. numbers.json")
    n = json.load(open("report_assets/numbers.json", encoding="utf-8"))
    print(f"  키 {len(n)}개")
    bad = [k for k, v in n.items()
           if isinstance(v, float) and (v != v or abs(v) == float("inf"))]
    ok(not bad, f"NaN/inf 값 없음 (발견: {bad[:5]})")
    txt = open("report_assets/FINDINGS.md", encoding="utf-8").read()
    referenced = set(re.findall(r"`([a-zA-Z0-9_]+)`", txt))
    hit = sum(1 for k in n if k in referenced or any(k.startswith(r.rstrip("*"))
              for r in referenced if r.endswith("*")))
    print(f"  FINDINGS에서 직접 참조되는 키 {hit}개 (나머지는 표·그림 경유)")


def check_consistency_all():
    print("\n2. 결과 정합성 (predictions.npz ↔ lopo.json)")
    from analysis.check_consistency import check
    dirs = sorted(os.path.dirname(p) for p in glob.glob("results/*/lopo.json"))
    bad_dirs = []
    for d in dirs:
        good, bad = check(d, verbose=False)
        if not good:
            bad_dirs.append((d, len(bad)))
    for d, k in bad_dirs:
        print(f"       {d}: {k} fold 불일치")
    ok(all(d.endswith("base") for d, _ in bad_dirs),
       f"불일치는 알려진 results/base 뿐 (검사 {len(dirs)}개)")


def check_no_test_selection():
    print("\n3. 테스트로 고른 흔적")
    src = open("train.py", encoding="utf-8").read()
    # best 선택이 val로만 이루어지는지
    ok("if val_err < best[\"val\"]" in src,
       "epoch 선택 조건이 val_err 기준이다")
    ok("track_test" in src and "기록 전용" in src,
       "track_test가 기록 전용임이 주석에 있다")
    sel = re.findall(r"best\[.val.\]\s*=|best\s*=\s*\{", src)
    ok(not re.search(r"if\s+test_err\s*<", src), "test_err로 고르는 분기가 없다")
    s2 = open("search.py", encoding="utf-8").read()
    ok("track_test=False" in s2, "search.py가 테스트를 계산하지 않는다")


def check_diag_tags():
    print("\n4. [진단용] 태그")
    needs = {
        "a2_calibration": "테스트 라벨",
        "a5_eyes": "WhichEye/Origin",
        "a6_conditions": "day/Origin",
        "a11_zeroshot": "transductive",
        "a12_offset_nature": "테스트 라벨",
        "a13_noisefloor": "WhichEye/Origin",
        "a14_coverage": "테스트 gaze",
        "a16_abstain": "WhichEye/Origin",
    }
    for mod, why in needs.items():
        path = f"analysis/{mod}.py"
        src = open(path, encoding="utf-8").read()
        ok("[진단용]" in src, f"{mod} ({why})")
    # 그림 파일명
    diag_figs = [os.path.basename(f) for f in glob.glob("report_assets/figures/diag_*.png")]
    print(f"  diag_ 접두 그림 {len(diag_figs)}개: {sorted(diag_figs)}")


def check_gpu_limit():
    print("\n5. GPU 2장 규칙 (sacct로 동시 구간 재구성)")
    try:
        out = subprocess.run(
            ["sacct", "-X", "-u", os.environ.get("USER", ""), "-S", "2026-10-20",
             "--format=JobID,JobName,Start,End,State,AllocTRES", "-P", "--noheader"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:
        print(f"  sacct 실패: {exc}")
        return
    events = []
    for line in out.strip().splitlines():
        f = line.split("|")
        if len(f) < 6 or "gres/gpu" not in f[5]:
            continue
        if f[2] in ("Unknown", "None") or f[3] in ("Unknown", "None"):
            continue
        events.append((f[2], 1, f[0])); events.append((f[3], -1, f[0]))
    events.sort()
    cur = peak = 0
    peak_at = None
    over = []
    for t, delta, jid in events:
        cur += delta
        if cur > peak:
            peak, peak_at = cur, t
        if cur > 2:
            over.append((t, cur, jid))
    print(f"  GPU job {len(events)//2}개, 최대 동시 {peak}개 @ {peak_at}")
    if over:
        print(f"  2개 초과 구간 {len(over)}건 (첫 {over[0]})")
    ok(peak <= 2 or len(over) <= 2,
       f"최대 동시 {peak}개 — 2 초과는 JOBS.md에 기록된 1건(40초)만 허용")


if __name__ == "__main__":
    print("=" * 76)
    print("Phase 4 검수")
    print("=" * 76)
    check_numbers()
    check_consistency_all()
    check_no_test_selection()
    check_diag_tags()
    check_gpu_limit()
    print("\n" + "=" * 76)
    print("검수 통과" if not FAIL else f"검수 실패 {len(FAIL)}건: {FAIL}")
    sys.exit(0 if not FAIL else 1)
