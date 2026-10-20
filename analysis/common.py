"""분석 공통 — 로딩, 통계, 그림 설정. CPU 전용."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset import PERSONS                     # noqa: E402
from metrics import angular_error               # noqa: E402

FIG = os.path.join(ROOT, "report_assets", "figures")
TAB = os.path.join(ROOT, "report_assets", "tables")
NUM = os.path.join(ROOT, "report_assets", "numbers.json")
DEG = np.degrees

plt.rcParams.update({
    "figure.dpi": 300, "savefig.bbox": "tight", "font.size": 8,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})


def use_korean_font() -> None:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"):
        if name in have:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return


use_korean_font()


# ----------------------------------------------------------------- 로딩

def load_run(run_dir: str) -> dict:
    """한 실행의 예측·메타를 읽는다. 정합성 검사를 통과하지 못하면 예외."""
    from analysis.check_consistency import check
    ok, bad = check(run_dir, verbose=False)
    if not ok:
        raise RuntimeError(f"{run_dir}: 정합성 실패 {len(bad)}개 fold. 분석에 쓰지 않는다.")

    with np.load(os.path.join(run_dir, "predictions.npz")) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.load(open(os.path.join(run_dir, "lopo.json"), encoding="utf-8"))
    d["err"] = angular_error(d["pred"], d["gaze"])
    d["fold_err"] = np.array([d["err"][d["person"] == p].mean() for p in PERSONS])
    d["dir"] = run_dir
    return d


def load_folds(run_dir: str, keys=None) -> dict:
    """fold npz들을 사람 순서로 읽어 dict of list로."""
    out = {}
    for p in PERSONS:
        f = os.path.join(run_dir, "folds", f"{p}.npz")
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            out[p] = {k: z[k] for k in (keys or z.files) if k in z.files}
    return out


# ----------------------------------------------------------------- 통계

def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """fold 간 대응표본 비교. a - b 를 본다 (a가 변형, b가 기준)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except ValueError:
        w, pw = float("nan"), float("nan")
    return {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float(d.mean()), "se": float(se),
            "t": float(t), "p": float(p), "wilcoxon_p": float(pw),
            "significant": bool(p < 0.05)}


def sig(x, n: int = 3) -> float:
    """유효숫자 n자리로 줄인다.

    p값에 round(x, 5)를 쓰면 1.2e-09가 0.0이 되고 7.1e-06이 1e-05가 된다.
    절대 규칙 6("반올림으로 의미를 바꾸지 않는다")에 걸리므로 이쪽을 쓴다.
    """
    x = float(x)
    if not np.isfinite(x) or x == 0.0:
        return x
    return float(f"{x:.{n}g}")


def fmt_paired(name: str, r: dict) -> str:
    mark = "유의" if r["significant"] else "구분 불가"
    return (f"{name:<28}{r['mean_a']:>8.3f}{r['delta']:>+9.3f}{r['se']:>8.3f}"
            f"{r['t']:>8.2f}{r['p']:>9.4f}  {mark}")


def pearson(x, y) -> tuple[float, float]:
    r, p = stats.pearsonr(np.asarray(x, float), np.asarray(y, float))
    return float(r), float(p)


def spearman(x, y) -> tuple[float, float]:
    r, p = stats.spearmanr(np.asarray(x, float), np.asarray(y, float))
    return float(r), float(p)


# ----------------------------------------------------------------- 출력

def save_fig(fig, name: str, data: dict | None = None) -> str:
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    if data is not None:
        import csv
        csv_path = path.rsplit(".", 1)[0] + ".csv"
        cols = list(data)
        rows = zip(*[np.asarray(data[c]).ravel() for c in cols])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    print(f"  → figures/{name}")
    return path


def save_table(name: str, header: list[str], rows: list[list]) -> None:
    os.makedirs(TAB, exist_ok=True)
    import csv
    with open(os.path.join(TAB, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(os.path.join(TAB, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(map(str, header)) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join("" if v is None else str(v) for v in r) + " |\n")
    print(f"  → tables/{name}.{{csv,md}}")


def put_numbers(**kv) -> None:
    os.makedirs(os.path.dirname(NUM), exist_ok=True)
    cur = json.load(open(NUM, encoding="utf-8")) if os.path.exists(NUM) else {}
    cur.update(kv)
    with open(NUM, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=2, ensure_ascii=False)


def section(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
