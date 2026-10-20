"""report_assets/ 전체를 다시 만드는 단일 진입점.

  python analysis/run_all.py            # 전부
  python analysis/run_all.py a1 a3      # 일부만

GPU를 쓰지 않는다. results/ 아래 저장된 npz·json만 읽는다.
"""
from __future__ import annotations

import os
import sys
import traceback

# `python analysis/run_all.py`로 직접 실행해도 되게 레포 루트를 경로에 넣는다
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

STEPS = [
    ("check", "결과 정합성 검사", "analysis.check_consistency", None),
    ("a1", "오차 분해 (사람 offset)", "analysis.a1_offset", "run"),
    ("a2", "few-shot calibration [진단용]", "analysis.a2_calibration", "run"),
    ("a3", "regression dilution", "analysis.a3_dilution", "run"),
    ("a4", "모델 선택 잡음", "analysis.a4_selection", "run"),
    ("a5", "좌우 눈 일관성 [진단용]", "analysis.a5_eyes", "run"),
    ("a6", "조건별 오차 [진단용]", "analysis.a6_conditions", "run"),
    ("a7", "fold 난이도 재현성", "analysis.a7_difficulty", "run"),
    ("a9", "처리량·효율", "analysis.a9_efficiency", "run"),
    ("a10", "라벨 없는 신호", "analysis.a10_labelfree", "run"),
    ("a11", "라벨 0장 재중심화 [진단용]", "analysis.a11_zeroshot", "run"),
    ("a12", "offset의 정체 [진단용]", "analysis.a12_offset_nature", "run"),
    ("a13", "오차 분산 3층 분해 [진단용]", "analysis.a13_noisefloor", "run"),
    ("a8", "해석 (occlusion·probe)", "analysis.a8_interpret", "run"),
    ("a14", "바깥쪽 시선: 밀도인가 성질인가 [진단용]", "analysis.a14_coverage", "run"),
    ("a15", "ensemble이 줄이는 성분", "analysis.a15_ensemble_mechanism", "run"),
    ("a16", "라벨 없는 신뢰도 신호 [진단용]", "analysis.a16_abstain", "run"),
    ("a17", "offset ↔ 눈 생김새 [진단용]", "analysis.a17_appearance", "run"),
    ("e2", "ensemble 표", "analysis.e2_ensemble", "run"),
    ("e3", "ablation 표", "analysis.e3_ablation", "run"),
    ("a18", "ablation Δ ↔ seed 잡음", "analysis.a18_ablation_null", "run"),
    ("a19", "문헌 수치와의 비교 가능성", "analysis.a19_literature_anchor", "run"),
    ("final", "최종 결과 표", "analysis.final_table", "run"),
    ("verify", "검수", "analysis.verify", None),
]


def main(argv: list[str]) -> int:
    want = set(argv) or {s[0] for s in STEPS}
    failed = []
    for key, label, mod, fn in STEPS:
        if key not in want:
            continue
        print(f"\n{'#' * 76}\n# {key}  {label}\n{'#' * 76}")
        try:
            m = __import__(mod, fromlist=["*"])
            if fn is None:
                import importlib
                importlib.reload(m)
                if hasattr(m, "check"):
                    for d in ("results/final", "results/p1_seed0"):
                        import os
                        if os.path.exists(f"{d}/lopo.json"):
                            m.check(d)
            else:
                getattr(m, fn)()
        except Exception:
            traceback.print_exc()
            failed.append(key)
    print(f"\n{'=' * 76}")
    print("전부 성공" if not failed else f"실패한 단계: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
