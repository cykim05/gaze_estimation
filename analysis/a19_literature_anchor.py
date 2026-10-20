"""A19 — 우리 수치를 문헌 수치와 직접 비교할 수 있는가.

Zhang et al.의 within-MPIIGaze LOPO 보고값(GazeNet 5.4°, MnistNet 5.5°)과 우리
5.68°를 나란히 놓고 싶어진다. 비교가 성립하는지 두 단계로 확인한다.

1. 평가 집합이 같은가 — gaze 분포를 논문이 본문에 적어 둔 범위와 맞춰 본다.
2. 난이도 앵커가 같은가 — '평균 gaze를 그냥 출력하는 예측기'의 오차를 맞춰 본다.
   이 값은 축 부호 규약과 무관하고, 모델과도 무관한 순수 데이터 속성이다.

1은 맞고 2는 어긋난다. 그 조합이 결론을 만든다. CPU 전용.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (PERSONS, angular_error, load_run, plt, put_numbers,
                             save_fig, save_table, section)

# Zhang et al., "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze
# Estimation", TPAMI 2019 (arXiv:1711.09017) — CVPR 2015 논문의 확장판.
# 아래 값은 Sec. V-A2(Cross-Person Evaluation) 본문에서 직접 인용한 것이고,
# 2026-10-20에 ar5iv 전문으로 확인했다. Fig. 12의 막대값은 본문에 없어 쓰지 않는다.
LIT = {
    "paper": "Zhang et al., TPAMI 2019 (arXiv:1711.09017), Sec. V-A2",
    "gazenet_deg": 5.4,      # "from a gaze estimation error of 5.4 ... degrees" (Sec. VI)
    "mnistnet_deg": 5.5,     # "the second-best MnistNet with 5.5 degrees mean error"
    "meanpred_deg": 13.9,    # "always outputs the average gaze direction ... 13.9 degrees"
    "yaw_range": (-18.0, 18.0),      # Sec. III-B, horizontal
    "pitch_range": (-1.5, 20.0),     # Sec. III-B, vertical (논문 부호 규약)
}


def lopo_mean_predictor(gaze: np.ndarray, person: np.ndarray) -> np.ndarray:
    """사람별로, '학습 14명의 평균 gaze'를 그대로 내놓는 예측기의 오차."""
    out = []
    for p in PERSONS:
        te = person == p
        mu = gaze[~te].mean(axis=0, keepdims=True)
        out.append(angular_error(np.repeat(mu, te.sum(), axis=0), gaze[te]).mean())
    return np.array(out)


def run() -> None:
    section("A19 — 문헌 수치와의 비교 가능성 [진단용 아님: 라벨 분포만 본다]")

    d = load_run("results/p1_seed0")
    gaze, person = d["gaze"], d["person"]
    yaw, pitch = np.degrees(gaze[:, 0]), np.degrees(gaze[:, 1])

    mp = lopo_mean_predictor(gaze, person)

    # final_table.py의 하드코딩 값(이전 EDA 단계에서 측정)과 독립 재계산을 맞춰 본다.
    from analysis.final_table import MEANPRED
    gap = np.abs(np.round(mp, 2) - np.array(MEANPRED)).max()
    print(f"\nfinal_table.MEANPRED 교차 확인: 최대 불일치 {gap:.3f}° "
          f"({'일치' if gap <= 0.005 else '불일치 — 확인 필요'})")
    assert gap <= 0.005, f"하드코딩된 하한선이 재계산과 다르다 ({gap:.3f}°)"

    print(f"\n평균 gaze 예측기(LOPO, 라벨에서 직접 재계산): {mp.mean():.3f}° "
          f"(사람별 {mp.min():.2f}~{mp.max():.2f}°)")
    print(f"같은 논문의 같은 예측기:                      {LIT['meanpred_deg']:.1f}°")
    print(f"  → 난이도 앵커가 {LIT['meanpred_deg'] - mp.mean():+.2f}° 다르다 "
          f"({LIT['meanpred_deg'] / mp.mean():.2f}배)")

    print(f"\n우리 평가 집합의 gaze 분포 (n={len(gaze):,})")
    for name, v, lit in (("yaw  ", yaw, LIT["yaw_range"]), ("pitch", pitch, LIT["pitch_range"])):
        lo, hi = np.percentile(v, [0.5, 99.5])
        print(f"  {name}  평균 {v.mean():+6.2f}°  표준편차 {v.std():5.2f}°  "
              f"1~99%p [{lo:+6.2f}, {hi:+6.2f}]  전체 [{v.min():+6.2f}, {v.max():+6.2f}]"
              f"   논문 보고 범위 [{lit[0]:+.1f}, {lit[1]:+.1f}]")

    # 논문은 vertical 범위를 [-1.5, +20]로 적었다. 우리 pitch는 [-19.1, +1.6]이다.
    # 부호만 뒤집으면 겹친다 — 즉 같은 분포를 반대 부호 규약으로 적은 것이다.
    pl, ph = np.percentile(pitch, [0.5, 99.5])
    flip_gap = max(abs(-ph - LIT["pitch_range"][0]), abs(-pl - LIT["pitch_range"][1]))
    yl, yh = np.percentile(yaw, [0.5, 99.5])
    yaw_gap = max(abs(yl - LIT["yaw_range"][0]), abs(yh - LIT["yaw_range"][1]))
    print(f"\n평가 집합 동일성 확인 (1~99%p 대 논문 본문 범위)")
    print(f"  yaw    최대 불일치 {yaw_gap:.2f}°")
    print(f"  pitch  부호를 뒤집으면 [{-ph:+.2f}, {-pl:+.2f}] — 최대 불일치 {flip_gap:.2f}°")
    print(f"  → 두 평가 집합의 gaze 분포는 {max(yaw_gap, flip_gap):.2f}° 안에서 일치한다. "
          f"같은 subset이다.")

    # 각거리 기준 분산 — 축 분해 없이 '퍼진 정도'를 본 값.
    mu_all = gaze.mean(axis=0, keepdims=True)
    spread = angular_error(np.repeat(mu_all, len(gaze), axis=0), gaze)
    print(f"\n전체 평균 방향으로부터의 각거리: 평균 {spread.mean():.3f}°, "
          f"중앙값 {np.median(spread):.3f}°, 90%p {np.percentile(spread, 90):.3f}°")

    rows = [
        ["평균 gaze 예측기 (LOPO)", f"{mp.mean():.3f}", f"{LIT['meanpred_deg']:.1f}",
         f"{LIT['meanpred_deg'] - mp.mean():+.2f}"],
        ["yaw 표준편차 (°)", f"{yaw.std():.2f}", "본문에 없음 (범위만)", "—"],
        ["pitch 표준편차 (°)", f"{pitch.std():.2f}", "본문에 없음 (범위만)", "—"],
        ["yaw 1~99%p (°)", f"[{np.percentile(yaw, 0.5):+.1f}, {np.percentile(yaw, 99.5):+.1f}]",
         f"[{LIT['yaw_range'][0]:+.1f}, {LIT['yaw_range'][1]:+.1f}]", "—"],
        ["pitch 1~99%p (°)", f"[{np.percentile(pitch, 0.5):+.1f}, {np.percentile(pitch, 99.5):+.1f}]",
         f"[{LIT['pitch_range'][0]:+.1f}, {LIT['pitch_range'][1]:+.1f}]", "—"],
    ]
    save_table("a19_literature_anchor", ["항목", "우리 평가 집합", LIT["paper"], "차이"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    axes[0].bar(["우리\n평균예측기", "논문\n평균예측기"], [mp.mean(), LIT["meanpred_deg"]],
                color=["#4477aa", "#bbbbbb"], width=.55)
    for i, v in enumerate([mp.mean(), LIT["meanpred_deg"]]):
        axes[0].text(i, v + .25, f"{v:.2f}°", ha="center", fontsize=7)
    axes[0].set(ylabel="LOPO 평균 각오차 (°)", ylim=(0, 16),
                title="같은 '아무것도 안 하는' 예측기의 오차")
    axes[1].hist2d(yaw, pitch, bins=60, cmap="magma")
    axes[1].set(xlabel="yaw (°)", ylabel="pitch (°)", title="우리 평가 집합의 gaze 분포")
    axes[1].grid(False)
    save_fig(fig, "32_literature_anchor.png",
             {"which": ["ours", "paper"], "meanpred_deg": [mp.mean(), LIT["meanpred_deg"]]})

    put_numbers(a19_meanpred_lopo_deg=round(float(mp.mean()), 3),
                a19_meanpred_paper_deg=LIT["meanpred_deg"],
                a19_anchor_gap_deg=round(float(LIT["meanpred_deg"] - mp.mean()), 3),
                a19_anchor_ratio=round(float(LIT["meanpred_deg"] / mp.mean()), 3),
                a19_yaw_sd_deg=round(float(yaw.std()), 3),
                a19_pitch_sd_deg=round(float(pitch.std()), 3),
                a19_spread_mean_deg=round(float(spread.mean()), 3),
                a19_paper_gazenet_deg=LIT["gazenet_deg"],
                a19_paper_mnistnet_deg=LIT["mnistnet_deg"])

    save_table("a19_subset_identity",
               ["축", "우리 1~99%p (°)", "논문 본문 범위 (°)", "최대 불일치 (°)"],
               [["yaw", f"[{yl:+.2f}, {yh:+.2f}]",
                 f"[{LIT['yaw_range'][0]:+.1f}, {LIT['yaw_range'][1]:+.1f}]", f"{yaw_gap:.2f}"],
                ["pitch (부호 뒤집어 맞춤)", f"[{-ph:+.2f}, {-pl:+.2f}]",
                 f"[{LIT['pitch_range'][0]:+.1f}, {LIT['pitch_range'][1]:+.1f}]",
                 f"{flip_gap:.2f}"]])

    put_numbers(a19_yaw_range_gap_deg=round(float(yaw_gap), 3),
                a19_pitch_range_gap_deg=round(float(flip_gap), 3))

    print(f"""
결론: 숫자를 나란히 놓지 않는다. 근거는 두 측정이 서로 어긋난다는 것이다.

  (1) 평가 집합은 같다. gaze 분포가 논문 본문의 범위와 {max(yaw_gap, flip_gap):.2f}° 안에서 겹친다
      (yaw {yaw_gap:.2f}°, pitch는 부호 규약만 반대이고 {flip_gap:.2f}°). 같은 1,500×2×15 = 45,000장이다.
  (2) 그런데 난이도 앵커가 안 맞는다. 모델과 무관하게 '평균 gaze를 출력'하기만 하는
      예측기의 LOPO 오차가 우리 데이터에서 {mp.mean():.2f}°인데, 논문은 같은 예측기를
      {LIT['meanpred_deg']:.1f}°로 적었다 ({LIT['meanpred_deg'] / mp.mean():.2f}배).

  (1)이 맞으면 (2)는 데이터 차이로 설명되지 않는다. 그리고 그 논문은 13.9°라는 같은 숫자를
  Abstract에서는 cross-dataset state of the art로, Sec. V-A2에서는 within-dataset 평균
  예측기로 쓴다 — 본문 Sec. V-A1이 cross-dataset 평균 예측기를 34.2°로 따로 적으므로
  둘 중 하나는 잘못 들어간 값이다. 즉 **문헌의 앵커를 우리 쪽에서 재현할 수 없다.**

  따라서 5.4° 대 5.68°는 비교로 쓰지 않는다. 우열을 주장할 수 없고, 주장할 필요도 없다.
  우리가 통제한 비교는 전부 한 평가 집합 안에 있다: 하한선 {mp.mean():.2f}° → ridge 7.27°
  → 단일 모델 5.94° → ensemble 5.68°. 문헌 수치는 '이 과제의 보고값이 5~6° 자리수'라는
  맥락으로만 인용하고, 그 옆에 이 앵커 불일치를 같이 적는다.""")


if __name__ == "__main__":
    run()
