"""오차 분석 — `python analyze.py` (train.py를 먼저 돌려 results/가 있어야 한다)

표에 찍힌 평균 하나로는 모델이 어디서 틀리는지 알 수 없다. 저장해 둔 예측을
다시 읽어, 오차가 어떤 샘플과 어떤 사람에게 몰리는지 본다. 학습은 다시 하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import dataset
from dataset import PERSONS
from eda import _use_korean_font, save, section
from metrics import angular_error, fold_table, fold_table_header

DEG = np.degrees


def load_predictions(out_dir: str):
    path = os.path.join(out_dir, "predictions.npz")
    with np.load(path) as z:
        person, pred, gaze, head = z["person"], z["pred"], z["gaze"], z["head"]
    with open(os.path.join(out_dir, "lopo.json")) as f:
        run = json.load(f)
    return person, pred, gaze, head, run


def describe_results(run, person, err) -> dict[str, float]:
    section("1. fold별 결과")

    errors = {r["test_person"]: r["test_error"] for r in run["results"]}
    print(fold_table_header())
    print(fold_table("GazeNet (numpy)", errors))

    values = np.array([errors[p] for p in PERSONS])
    print(f"\n평균 {values.mean():.2f}°  표준편차 {values.std():.2f}°  "
          f"최고 {values.min():.2f}° ({PERSONS[values.argmin()]})  "
          f"최저 {values.max():.2f}° ({PERSONS[values.argmax()]})")

    print(f"\n선택된 epoch: " + ", ".join(
        f"{r['test_person']}={r['best_epoch']}" for r in sorted(
            run["results"], key=lambda r: r["test_person"])))
    final = np.array([r["final_test_error"] for r in run["results"]])
    chosen = np.array([r["test_error"] for r in run["results"]])
    print(f"마지막 epoch로 그냥 끝냈다면 {final.mean():.2f}°, "
          f"검증으로 고른 결과는 {chosen.mean():.2f}° "
          f"({final.mean() - chosen.mean():+.2f}°)")
    return errors


def describe_error_shape(err) -> None:
    section("2. 오차의 모양 — 평균 뒤에 가려진 것")

    qs = [50, 75, 90, 95, 99]
    print(f"평균 {err.mean():.2f}°, 중앙값 {np.median(err):.2f}°")
    print("분위수: " + "  ".join(f"{q}%={np.percentile(err, q):.1f}°" for q in qs))
    tail = err >= np.percentile(err, 90)
    print(f"→ 상위 10% 샘플이 전체 오차 합의 {err[tail].sum() / err.sum() * 100:.0f}%를 낸다. "
          "분포가 길게 끌린다.")
    print(f"  오차 5° 이하 {100 * (err <= 5).mean():.1f}%, "
          f"10° 초과 {100 * (err > 10).mean():.1f}%, "
          f"20° 초과 {100 * (err > 20).mean():.1f}%")


def describe_drivers(samples, person, gaze, head, err) -> None:
    section("3. 오차는 어디에 몰리나")

    # 예측은 fold를 사람 순으로 이어 붙였고 dataset.load()도 같은 순서다.
    # 아래 지표들은 samples에서 뽑으므로 두 순서가 같아야 한다.
    assert np.array_equal(np.asarray(person), samples.person_array), \
        "예측 순서와 dataset.load() 순서가 어긋난다"

    img = samples.image[:, 0].astype(np.float32)
    band_var = img[:, 12:24, 15:45].var(axis=(1, 2))      # 눈이 감겼는지
    head_mag = DEG(np.linalg.norm(head, axis=1))          # 고개를 얼마나 돌렸나
    gaze_mag = DEG(np.linalg.norm(gaze, axis=1))          # 시선이 얼마나 치우쳤나

    for name, value, unit in [("눈 영역 분산", band_var, ""),
                              ("head pose 크기", head_mag, "°"),
                              ("gaze 크기", gaze_mag, "°")]:
        edges = np.percentile(value, [0, 20, 40, 60, 80, 100])
        print(f"\n{name}별 5분위 평균 오차")
        for i in range(5):
            m = (value >= edges[i]) & (value <= edges[i + 1])
            print(f"  {edges[i]:7.1f}{unit} ~ {edges[i + 1]:7.1f}{unit}  "
                  f"n={m.sum():5d}  오차 {err[m].mean():.2f}°")
        r = np.corrcoef(value, err)[0, 1]
        print(f"  상관계수 r = {r:+.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, (name, value, unit) in zip(axes, [("눈 영역 분산", band_var, ""),
                                              ("head pose 크기 (°)", head_mag, ""),
                                              ("gaze 크기 (°)", gaze_mag, "")]):
        ax.hexbin(value, err, gridsize=40, cmap="magma", bins="log")
        ax.set(xlabel=name, ylabel="각도 오차 (°)")
    fig.suptitle("오차를 끌어올리는 요인", fontsize=9)
    save(fig, "08_error_drivers.png")


def describe_appearance_gap(samples, errors) -> None:
    section("4. EDA 가설 검증 — 외모가 낯설수록 어려운가")

    means = np.stack([samples.only(p).image[:, 0].mean(0).ravel() for p in PERSONS])
    means = means.astype(np.float64)

    gap, err = [], []
    for k, p in enumerate(PERSONS):
        others = np.delete(means, k, axis=0).mean(0)       # 그 fold의 학습 14명 평균
        gap.append(np.linalg.norm(means[k] - others) / np.sqrt(means.shape[1]))
        err.append(errors[p])
    gap, err = np.array(gap), np.array(err)
    r = np.corrcoef(gap, err)[0, 1]

    print("사람별 '학습 14명 평균 눈과의 거리'와 테스트 오차\n")
    print(f"{'person':>7}{'거리(/255)':>12}{'오차(°)':>10}")
    for p, g, e in sorted(zip(PERSONS, gap, err), key=lambda t: -t[1]):
        print(f"{p:>7}{g:>12.1f}{e:>10.2f}")
    print(f"\n상관계수 r = {r:+.3f}")
    print("→ " + ("외모가 학습 분포에서 멀수록 오차가 크다. EDA 3·5절의 가설과 맞는다."
                  if r > 0.3 else
                  "평균 눈 이미지 거리만으로는 fold 난이도가 설명되지 않는다. "
                  "외모 차이는 평균 이미지보다 세밀한 수준에서 작동한다."))

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(gap, err, s=28, c="#1f4e79")
    for p, g, e in zip(PERSONS, gap, err):
        ax.annotate(p, (g, e), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="학습 14명 평균 눈과의 거리 (/255)", ylabel="테스트 각도 오차 (°)",
           title=f"r = {r:+.3f}")
    save(fig, "09_appearance_gap.png")


def show_worst(samples, pred, gaze, err, n=12) -> None:
    section("5. 가장 많이 틀린 샘플")

    worst = np.argsort(-err)[:n]
    best = np.argsort(err)[:n]
    print(f"최악 {n}개 오차: {err[worst].min():.1f}~{err[worst].max():.1f}°")
    print(f"최선 {n}개 오차: {err[best].min():.2f}~{err[best].max():.2f}°")

    fig, axes = plt.subplots(2, n, figsize=(n, 2.8))
    for ax, i in zip(axes[0], worst):
        ax.imshow(samples.image[i, 0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{err[i]:.0f}°", fontsize=6)
    for ax, i in zip(axes[1], best):
        ax.imshow(samples.image[i, 0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{err[i]:.1f}°", fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_ylabel("최악", fontsize=7)
    axes[1, 0].set_ylabel("최선", fontsize=7)
    save(fig, "10_worst_best.png")


def describe_bias(pred, gaze, err) -> None:
    section("6. 체계적인 치우침이 있나")

    resid = DEG(pred - gaze)
    print(f"잔차 평균: yaw {resid[:, 0].mean():+.2f}°, pitch {resid[:, 1].mean():+.2f}°")
    print(f"잔차 표준편차: yaw {resid[:, 0].std():.2f}°, pitch {resid[:, 1].std():.2f}°")

    # 예측이 평균 쪽으로 쏠리는지(회귀 모델의 전형적인 수축)
    for k, name in enumerate(("yaw", "pitch")):
        slope = np.polyfit(DEG(gaze[:, k]), DEG(pred[:, k]), 1)[0]
        print(f"{name}: 정답 1° 변화에 예측 {slope:.2f}° 변화 "
              f"(1.00이면 수축 없음)")
    print("→ 기울기가 1보다 작으면 예측이 평균 쪽으로 수축한 것이다. "
          "극단적인 시선에서 오차가 커지는 원인이다.")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, k, name in zip(axes[:2], (0, 1), ("yaw", "pitch")):
        ax.hexbin(DEG(gaze[:, k]), DEG(pred[:, k]), gridsize=50, cmap="magma", bins="log")
        lim = [DEG(gaze[:, k]).min(), DEG(gaze[:, k]).max()]
        ax.plot(lim, lim, "w--", lw=0.8)
        ax.set(xlabel=f"정답 {name} (°)", ylabel=f"예측 {name} (°)", title=name)
    axes[2].hist(err, bins=80, color="#444")
    axes[2].set(xlabel="각도 오차 (°)", ylabel="샘플 수",
                title=f"평균 {err.mean():.2f}° / 중앙값 {np.median(err):.2f}°")
    save(fig, "11_predictions.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/base")
    args = p.parse_args()

    _use_korean_font()
    os.makedirs("figures", exist_ok=True)

    person, pred, gaze, head, run = load_predictions(args.out)
    samples = dataset.load(verbose=False)
    err = angular_error(pred, gaze)

    errors = describe_results(run, person, err)
    describe_error_shape(err)
    describe_drivers(samples, person, gaze, head, err)
    describe_appearance_gap(samples, errors)
    show_worst(samples, pred, gaze, err)
    describe_bias(pred, gaze, err)
    print()


if __name__ == "__main__":
    main()
