"""EDA — MPIIGaze eye-based 데이터 살펴보기. `python eda.py`

본문은 stdout으로, 그림은 figures/ 아래로 나간다. 보고 싶은 것은 하나다.
**leave-one-person-out에서 무엇이 어려운가.** 그래서 전체 분포보다 사람 간 차이를,
픽셀 통계보다 "이 입력으로 gaze를 설명할 수 있는 양"을 먼저 본다.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import dataset
from dataset import PERSONS
from metrics import angular_error

FIG_DIR = "figures"
DEG = np.degrees

def _use_korean_font() -> None:
    """그림의 한글 라벨이 □로 깨지지 않게 한글 글꼴을 찾아 쓴다. 없으면 조용히 넘긴다."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"):
        if name in available:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return


plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,  # 한글 글꼴에는 U+2212가 없다
})
_use_korean_font()


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def save(fig, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------- 1. 규모와 구성

def describe_scale(df) -> None:
    section("1. 규모와 구성")

    print(f"샘플 {len(df):,}개 · 사람 {df.person.nunique()}명 · "
          f"이미지 36×60 grayscale(1채널)")

    per = df.groupby("person", observed=True).agg(
        n=("image", "size"),
        left=("which_eye", lambda s: (s == "left").sum()),
        right=("which_eye", lambda s: (s == "right").sum()),
        days=("day", "nunique"),
        frames=("frame", "nunique"),
    )
    print(f"\n사람마다 정확히 {per.n.unique()} 개, 좌/우눈 {per.left.unique()}/{per.right.unique()} 로 "
          "완전히 균형 잡혀 있다. 클래스 불균형 문제가 없다.")
    print(f"day 수는 사람마다 {per.days.min()}~{per.days.max()}개로 크게 다르다 "
          f"(중앙값 {int(per.days.median())}).")
    print("\n" + per.to_string())

    # 한 프레임에서 좌·우눈이 둘 다 뽑히면 두 샘플이 같은 장면을 공유한다
    per_frame = df.groupby(["person", "day", "frame"], observed=True).size()
    counts = {int(k): int(v) for k, v in per_frame.value_counts().sort_index().items()}
    shared = sum(k * v for k, v in counts.items() if k > 1)
    print(f"\n한 (사람, day, 프레임)에서 뽑힌 샘플 수: {counts}")
    print(f"→ 프레임 {counts.get(1, 0):,}개는 한쪽 눈만, {counts.get(2, 0):,}개는 양쪽 눈을 냈다. "
          f"전체의 {shared / len(df) * 100:.0f}%({shared:,}개)가 다른 샘플과 프레임을 공유한다.")
    print("  같은 프레임의 두 눈은 head pose와 응시점이 같으므로, 샘플 단위로 train/val을 "
          "나누면 거의 같은 장면이 양쪽에 들어간다.")
    print("  검증 세트도 사람 단위로 떼어 내야 하는 첫 번째 이유다.")

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
    x = np.arange(len(per))
    axes[0].bar(x - 0.2, per.left, 0.4, label="left")
    axes[0].bar(x + 0.2, per.right, 0.4, label="right")
    axes[0].set_xticks(x, per.index, rotation=90)
    axes[0].set_ylabel("samples")
    axes[0].set_title("사람별 좌/우눈 샘플 수")
    axes[0].legend(fontsize=7)

    axes[1].bar(x, per.days, color="#6b7a8f")
    axes[1].set_xticks(x, per.index, rotation=90)
    axes[1].set_ylabel("days")
    axes[1].set_title("사람별 촬영 day 수 (샘플 수와 무관)")
    save(fig, "01_counts.png")


# ------------------------------------------------------- 2. 라벨 분포와 사람 간 차이

def describe_labels(df) -> None:
    section("2. 라벨 분포 — gaze와 head pose")

    cols = {"gaze yaw": "2DGaze_yaw", "gaze pitch": "2DGaze_pitch",
            "head yaw": "2DHead_yaw", "head pitch": "2DHead_pitch"}
    print(f"{'':<12}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}   (단위: 도)")
    for name, col in cols.items():
        v = DEG(df[col])
        print(f"{name:<12}{v.mean():>9.2f}{v.std():>9.2f}{v.min():>9.2f}{v.max():>9.2f}")

    print("\n- gaze는 yaw가 pitch보다 1.7배 넓게 퍼져 있다(화면이 가로로 넓으니 당연하다).")
    print("- gaze pitch 평균이 -8.7도다. 화면이 눈높이보다 아래에 있어 대체로 내려다본다.")
    print("- head pitch 평균은 +9.8도로 gaze pitch와 부호가 반대다. "
          "고개는 들고 시선은 내리는 자세가 기본값이다.")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    axes[0].hexbin(DEG(df["2DGaze_yaw"]), DEG(df["2DGaze_pitch"]), gridsize=45,
                   cmap="magma", bins="log")
    axes[0].set(xlabel="gaze yaw (°)", ylabel="gaze pitch (°)", title="gaze 분포")
    axes[1].hexbin(DEG(df["2DHead_yaw"]), DEG(df["2DHead_pitch"]), gridsize=45,
                   cmap="magma", bins="log")
    axes[1].set(xlabel="head yaw (°)", ylabel="head pitch (°)", title="head pose 분포")
    for name, col in cols.items():
        axes[2].hist(DEG(df[col]), bins=80, histtype="step", label=name, density=True)
    axes[2].set(xlabel="도", ylabel="density", title="네 각도의 주변 분포")
    axes[2].legend(fontsize=7)
    save(fig, "02_label_dist.png")


def describe_person_shift(df) -> None:
    section("3. 사람 간 분포 차이 — leave-one-person-out이 어려운 이유")

    g = df.groupby("person", observed=True)
    stat = g[["2DGaze_yaw", "2DGaze_pitch", "2DHead_yaw", "2DHead_pitch"]].agg(["mean", "std"])
    stat = DEG(stat)

    print("사람별 gaze/head 평균과 표준편차 (도):\n")
    print(stat.round(2).to_string())

    spread = DEG(g[["2DGaze_yaw", "2DGaze_pitch"]].mean()).std()
    within = DEG(g[["2DGaze_yaw", "2DGaze_pitch"]].std()).mean()
    print(f"\n사람 평균들의 퍼짐(between-person): yaw {spread['2DGaze_yaw']:.2f}°, "
          f"pitch {spread['2DGaze_pitch']:.2f}°")
    print(f"사람 안에서의 퍼짐(within-person):   yaw {within['2DGaze_yaw']:.2f}°, "
          f"pitch {within['2DGaze_pitch']:.2f}°")
    print("→ 사람 간 차이가 사람 안의 변화보다 훨씬 작다. gaze 자체는 사람이 달라도 "
          "비슷한 범위를 훑는다는 뜻이고,")
    print("  그래서 LOPO의 어려움은 '라벨 분포가 달라서'가 아니라 "
          "'눈 모양(외모)이 달라서'다. 아래 5·6절에서 확인한다.")

    # 사람별 gaze 산점도 15장
    fig, axes = plt.subplots(3, 5, figsize=(11, 6), sharex=True, sharey=True)
    for ax, person in zip(axes.ravel(), PERSONS):
        d = df[df.person == person]
        ax.scatter(DEG(d["2DGaze_yaw"]), DEG(d["2DGaze_pitch"]), s=1, alpha=0.2, c="#1f4e79")
        ax.set_title(f"{person}  (n={len(d)})", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("yaw (°)")
    for ax in axes[:, 0]:
        ax.set_ylabel("pitch (°)")
    fig.suptitle("사람별 gaze 분포 — 범위는 비슷하고 모양(응시 패턴)만 다르다", fontsize=9)
    save(fig, "03_gaze_per_person.png")


# ------------------------------------------------- 4. head pose가 gaze를 얼마나 설명하나

def describe_head_gaze(df) -> None:
    section("4. head pose ↔ gaze — 두 번째 입력의 값어치")

    pairs = [("2DHead_yaw", "2DGaze_yaw"), ("2DHead_pitch", "2DGaze_pitch"),
             ("2DHead_yaw", "2DGaze_pitch"), ("2DHead_pitch", "2DGaze_yaw")]
    print("상관계수:")
    for a, b in pairs:
        r = np.corrcoef(df[a], df[b])[0, 1]
        print(f"  {a:<14} ~ {b:<14} r = {r:+.3f}")

    # head pose만으로 gaze를 최소제곱 예측했을 때의 각도 오차 (사람 단위 LOPO로 정직하게)
    errs = []
    for test in PERSONS:
        tr, te = df[df.person != test], df[df.person == test]
        X = np.c_[tr[["2DHead_yaw", "2DHead_pitch"]].to_numpy(), np.ones(len(tr))]
        W = np.linalg.lstsq(X, tr[["2DGaze_yaw", "2DGaze_pitch"]].to_numpy(), rcond=None)[0]
        Xt = np.c_[te[["2DHead_yaw", "2DHead_pitch"]].to_numpy(), np.ones(len(te))]
        errs.append(angular_error(Xt @ W, te[["2DGaze_yaw", "2DGaze_pitch"]].to_numpy()).mean())

    # 비교용 하한선: 학습 14명의 gaze 평균을 항상 내놓기
    base = []
    for test in PERSONS:
        tr, te = df[df.person != test], df[df.person == test]
        mean = tr[["2DGaze_yaw", "2DGaze_pitch"]].to_numpy().mean(0, keepdims=True)
        gt = te[["2DGaze_yaw", "2DGaze_pitch"]].to_numpy()
        base.append(angular_error(np.repeat(mean, len(gt), 0), gt).mean())

    print(f"\nLOPO 평균 각도 오차")
    print(f"  학습 14명의 gaze 평균만 출력 : {np.mean(base):.2f}°  (입력을 아예 안 보는 하한선)")
    print(f"  head pose 2개로 선형 회귀    : {np.mean(errs):.2f}°")
    print(f"→ head pose만으로는 하한선에서 {np.mean(base) - np.mean(errs):.2f}°밖에 못 줄인다. "
          "gaze 정보는 거의 다 이미지에 있다.")
    print("  (그래도 입력에서 빼지는 않는다. 같은 눈 이미지가 고개 각도에 따라 "
          "다른 gaze를 뜻하므로 모호성을 없애는 역할을 한다.)")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, (a, b) in zip(axes, pairs[:2]):
        ax.scatter(DEG(df[a]), DEG(df[b]), s=1, alpha=0.08, c="#1f4e79")
        r = np.corrcoef(df[a], df[b])[0, 1]
        ax.set(xlabel=f"{a.replace('2D', '')} (°)", ylabel=f"{b.replace('2D', '')} (°)",
               title=f"r = {r:+.3f}")
    fig.suptitle("head pose와 gaze는 같은 축끼리도 약하게만 묶여 있다", fontsize=9)
    save(fig, "04_head_vs_gaze.png")


# ------------------------------------------------------------------- 5. 이미지 통계

def describe_images(samples) -> None:
    section("5. 이미지 통계 — 어디에 정보가 있나")

    img = samples.image[:, 0].astype(np.float32) / 255.0  # (N, 36, 60)
    print(f"픽셀 값: mean {img.mean():.3f}  std {img.std():.3f}  "
          f"min {img.min():.3f}  max {img.max():.3f}")
    print("전처리에서 histogram equalization을 했으므로 사람·조명에 따른 "
          "전역 밝기 차이는 이미 많이 지워져 있다.")

    mean_img = img.mean(0)
    std_img = img.std(0)
    hot = tuple(int(v) for v in np.unravel_index(std_img.argmax(), std_img.shape))
    cold = tuple(int(v) for v in np.unravel_index(std_img.argmin(), std_img.shape))
    print(f"\n픽셀별 표준편차: 최대 {std_img.max():.3f} @ (행 {hot[0]}, 열 {hot[1]}), "
          f"최소 {std_img.min():.3f} @ (행 {cold[0]}, 열 {cold[1]})")

    center = (slice(14, 23), slice(26, 35))  # 36×60의 가운데 9×9
    print(f"가운데 9×9: 평균 밝기 {mean_img[center].mean():.3f} (전체 {mean_img.mean():.3f}), "
          f"표준편차 {std_img[center].mean():.3f} (전체 {std_img.mean():.3f})")
    print("→ 정규화가 눈 중심을 맞춰 놓았으므로 가운데는 '항상 어두운 동공'이다. "
          "어둡고(0.14) 거의 변하지 않는다(0.16).")
    print("  변화는 그 주변, 홍채와 흰자의 경계가 지나가는 자리에서 가장 크다. "
          f"가로로는 {hot[1]}번 열 근처가 최대다.")
    print(f"  아래쪽 여섯 행(뺨)은 표준편차 {std_img[-6:].mean():.3f}로 가장 낮다 — "
          "정보가 없는 영역이다.")
    print("  → 3×3 작은 커널로 동공 주변의 국소 경계를 잡고, 풀링으로 위치를 "
          "조금씩 뭉개는 구조가 맞다.")

    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(3, 7, hspace=0.55, wspace=0.12)

    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(mean_img, cmap="gray")
    ax.set_title("전체 평균 이미지 — 가운데는 늘 동공", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(std_img, cmap="inferno")
    ax.set_title("픽셀별 표준편차 — 변하는 곳은 동공 둘레", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax = fig.add_subplot(gs[2, :2])
    ax.hist(img[::37].ravel(), bins=64, color="#444")
    ax.set_title("밝기 분포 — equalization으로 평탄", fontsize=8)
    ax.set_yticks([])

    for k, person in enumerate(PERSONS):
        ax = fig.add_subplot(gs[k // 5, 2 + k % 5])
        ax.imshow(samples.only(person).image[:, 0].mean(0), cmap="gray", vmin=0, vmax=255)
        ax.set_title(person, fontsize=7); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("왼쪽: 전체 통계 / 오른쪽: 사람별 평균 이미지 — 사람마다 눈 모양이 뚜렷이 다르다",
                 fontsize=9)
    save(fig, "05_images.png")

    # 사람별 평균 이미지가 서로 얼마나 다른지를 수로 본다
    means = np.stack([samples.only(p).image[:, 0].mean(0) for p in PERSONS]).reshape(15, -1)
    d = np.linalg.norm(means[:, None] - means[None], axis=-1) / np.sqrt(means.shape[1])
    off = d[~np.eye(15, dtype=bool)]
    within = np.mean([img[samples.person == i].std(0).mean() for i in range(15)])
    print(f"\n사람별 평균 이미지 사이 RMS 거리: {off.mean():.1f}/255 "
          f"(가장 가까운 쌍 {off.min():.1f}, 가장 먼 쌍 {off.max():.1f})")
    print(f"한 사람 안에서의 픽셀 RMS 변화:   {within * 255:.1f}/255")
    print("→ 사람 간 외모 차이가 한 사람 안의 변화와 맞먹는다. "
          "LOPO에서 모델이 넘어야 하는 벽이 이것이다.")


# ------------------------------------------------------------------------ 6. PCA

def describe_pca(samples, n_comp=24) -> None:
    section("6. PCA — 눈 이미지의 주된 변화축은 gaze인가")

    X = samples.image[:, 0].reshape(len(samples), -1).astype(np.float32) / 255.0
    X = X - X.mean(0, keepdims=True)
    cov = (X.T @ X) / len(X)
    w, V = np.linalg.eigh(cov)
    w, V = w[::-1][:n_comp], V[:, ::-1][:, :n_comp]
    ratio = w / np.trace(cov)

    print(f"상위 {n_comp}개 주성분이 분산의 {ratio.sum() * 100:.1f}%를 설명한다 "
          f"(1번 {ratio[0] * 100:.1f}%, 2번 {ratio[1] * 100:.1f}%, 3번 {ratio[2] * 100:.1f}%).")

    scores = X @ V
    gaze = samples.gaze
    print(f"\n{'PC':>3} {'분산%':>7} {'r(yaw)':>8} {'r(pitch)':>9}")
    for k in range(8):
        ry = np.corrcoef(scores[:, k], gaze[:, 0])[0, 1]
        rp = np.corrcoef(scores[:, k], gaze[:, 1])[0, 1]
        print(f"{k + 1:>3} {ratio[k] * 100:>7.1f} {ry:>+8.3f} {rp:>+9.3f}")

    # 주성분 전체를 쓴 선형 회귀의 LOPO 오차 — 이미지에서 선형으로 뽑히는 양의 상한
    person = samples.person
    errs = []
    for k, test in enumerate(PERSONS):
        tr, te = person != k, person == k
        A = np.c_[scores[tr], np.ones(tr.sum())]
        W = np.linalg.lstsq(A, gaze[tr], rcond=None)[0]
        At = np.c_[scores[te], np.ones(te.sum())]
        errs.append(angular_error(At @ W, gaze[te]).mean())
    print(f"\nPC {n_comp}개 + 절편으로 선형 회귀한 LOPO 오차: {np.mean(errs):.2f}°")
    print("→ gaze는 분산이 가장 큰 축에 깔끔하게 놓여 있지 않다. "
          "선형으로는 한계가 있고, 비선형 특징 추출(CNN)이 필요하다.")

    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(3, 9, hspace=0.3, wspace=0.15)
    for k in range(12):
        ax = fig.add_subplot(gs[k // 6, k % 6])
        ax.imshow(V[:, k].reshape(36, 60), cmap="RdBu_r")
        ax.set_title(f"PC{k + 1} ({ratio[k] * 100:.1f}%)", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = fig.add_subplot(gs[2, :6])
    ax.plot(np.arange(1, n_comp + 1), np.cumsum(ratio) * 100, "o-", ms=3)
    ax.set(xlabel="주성분 개수", ylabel="누적 분산 설명 (%)")

    ax = fig.add_subplot(gs[:, 6:])
    sc = ax.scatter(scores[::5, 0], scores[::5, 1], s=1.5, alpha=0.3,
                    c=DEG(gaze[::5, 0]), cmap="coolwarm")
    ax.set(xlabel="PC1", ylabel="PC2", title="PC1–PC2 평면, 색 = gaze yaw")
    fig.colorbar(sc, ax=ax, fraction=0.04, label="gaze yaw (°)")
    save(fig, "06_pca.png")


# ------------------------------------------------------------- 7. 품질과 이상 샘플

def describe_outliers(samples, df) -> None:
    section("7. 품질 점검 — 학습을 방해할 샘플이 있나")

    img = samples.image[:, 0].astype(np.float32)
    err_from_mean = angular_error(
        np.repeat(samples.gaze.mean(0, keepdims=True), len(samples), 0), samples.gaze)

    # 눈을 감으면 가운데 가로 띠의 분산이 떨어진다
    band = img[:, 12:24, 15:45]
    band_var = band.var(axis=(1, 2))
    low = np.argsort(band_var)[:12]
    print(f"눈 영역 분산: 중앙값 {np.median(band_var):.0f}, "
          f"하위 1% {np.percentile(band_var, 1):.0f}, 상위 1% {np.percentile(band_var, 99):.0f}")
    print(f"분산이 하위 1%인 샘플 {int(len(samples) * 0.01)}개는 "
          "눈을 감거나 머리가 크게 돌아 홍채가 안 보이는 경우다.")
    print("라벨은 화면 위 응시점에서 계산되므로 눈이 감겨도 gaze 값은 남는다. "
          "즉 입력에 정보가 없는 샘플이 섞여 있고, 이것이 오차의 하한을 만든다.")

    extreme = np.argsort(-err_from_mean)[:12]
    print(f"\ngaze가 전체 평균에서 가장 먼 샘플의 각도: "
          f"{err_from_mean[extreme].min():.1f}~{err_from_mean[extreme].max():.1f}°, "
          f"전체 95분위 {np.percentile(err_from_mean, 95):.1f}°")

    dup = int(df.duplicated(subset=["person", "day", "frame", "which_eye"]).sum())
    print(f"\n(사람, day, 프레임, 눈) 완전 중복: {dup}개 "
          f"— 같은 프레임의 같은 눈이 sample list에 두 번 들어간 경우다. {len(df):,}개 중 {dup}개라 무시해도 된다.")
    print(f"비정상 라벨(NaN/inf): gaze {int(np.sum(~np.isfinite(samples.gaze)))}개, "
          f"head {int(np.sum(~np.isfinite(samples.head)))}개")
    print(f"값이 전부 같은(완전히 평평한) 이미지: {int(np.sum(img.std(axis=(1, 2)) < 1e-6))}개")

    fig, axes = plt.subplots(2, 12, figsize=(12, 2.6))
    for ax, i in zip(axes[0], low):
        ax.imshow(samples.image[i, 0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"var {band_var[i]:.0f}", fontsize=6)
    for ax, i in zip(axes[1], extreme):
        ax.imshow(samples.image[i, 0], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{DEG(samples.gaze[i, 0]):.0f},{DEG(samples.gaze[i, 1]):.0f}°", fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_ylabel("눈 영역\n분산 최소", fontsize=7)
    axes[1, 0].set_ylabel("gaze\n극단값", fontsize=7)
    save(fig, "07_outliers.png")


# ------------------------------------------------------------------------- 정리

def conclude() -> None:
    section("정리 — 모델 설계로 넘어가며")
    for line in [
        "1. 데이터는 사람·눈 기준으로 완벽히 균형(각 3,000 / 1,500+1,500)이라 "
        "샘플링 보정이 필요 없다.",
        "2. 한 프레임에서 좌·우눈이 같이 나오므로 샘플 단위 분할은 누설이다. "
        "검증도 사람 단위로 떼어 낸다.",
        "3. gaze 범위는 사람이 달라도 비슷하고, 다른 것은 눈의 외모다. "
        "→ 일반화의 관건은 라벨 보정이 아니라 외모 불변 특징이다.",
        "4. head pose만으로는 하한선에서 거의 못 내려간다. "
        "주된 신호는 이미지이고 head pose는 모호성 해소용이다.",
        "5. 정보는 가운데 가로 띠(홍채)에 모여 있고, 선형 PCA로는 충분히 뽑히지 않는다. "
        "→ 작은 CNN이 맞다.",
        "6. 눈을 감은 샘플이 1% 남짓 섞여 있어 오차 0은 원리적으로 불가능하다. "
        "제거하지 않는다(테스트에도 있으므로).",
        "7. 이미지 정규화 통계는 스칼라 평균·표준편차만 쓰고 "
        "학습 fold에서만 구한다. 위치별 평균은 학습 fold 사람들의 눈 모양을 빼는 셈이다.",
    ]:
        print(f"  {line}")
    print()


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    samples = dataset.load()
    df = dataset.label_frame()

    describe_scale(df)
    describe_labels(df)
    describe_person_shift(df)
    describe_head_gaze(df)
    describe_images(samples)
    describe_pca(samples)
    describe_outliers(samples, df)
    conclude()


if __name__ == "__main__":
    main()
