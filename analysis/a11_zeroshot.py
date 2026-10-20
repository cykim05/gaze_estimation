"""A11 — 라벨 0장 재중심화. **[진단용 — transductive]** (자율 탐색)

A1: 사람별 offset이 오차의 17%. A2: 라벨 10장이면 75% 회수.
**라벨을 한 장도 안 쓰고** offset을 지울 수 있는가?

착안: EDA 3.2에서 사람이 달라도 gaze 분포의 **평균**이 거의 같았다
(사람 평균들의 퍼짐 yaw 0.25° / pitch 1.04°, 사람 안 퍼짐 8.7° / 5.1°).
그렇다면 새 사람의 예측 평균이 학습 사람들의 gaze 평균에서 벗어난 만큼이
곧 그 사람에 대한 모델의 치우침이다.

    shift(사람) = mean(예측)  −  mean(학습 12명의 gaze)
                  └ 테스트 라벨 불필요    └ 학습 통계 (합법)

**왜 [진단용]인가**: 테스트 사람의 이미지 묶음으로 통계를 내므로 transductive다.
README의 "테스트 사람의 데이터는 정규화 통계 계산에 쓰지 않는다"에 걸린다.
메인 LOPO 수치에는 넣지 않고, 무엇이 가능한지를 보이는 용도로만 쓴다.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, angular_error, load_run, paired, pearson,
                             plt, put_numbers, save_fig, save_table, section)

NS = [10, 30, 100, 300, 1000, 3000]
BOOT = 100


def run(run_dir="results/final", tag="runB"):
    section(f"A11 — 라벨 0장 재중심화 [진단용·transductive] ({run_dir})")
    import dataset
    from train import split_persons

    d = load_run(run_dir)
    person, pred, gaze, err = d["person"], d["pred"], d["gaze"], d["err"]
    samples = dataset.load(verbose=False)

    mu_train = {}
    for p in PERSONS:
        tr_p, _ = split_persons(p, 2, list(PERSONS))
        mu_train[p] = samples.only(*tr_p).gaze.mean(0)

    # 전량 사용
    corrected = pred.copy()
    shifts = {}
    for p in PERSONS:
        m = person == p
        shifts[p] = pred[m].mean(0) - mu_train[p]
        corrected[m] = pred[m] - shifts[p]
    err_z = angular_error(corrected, gaze)
    f0 = d["fold_err"]
    f1 = np.array([err_z[person == p].mean() for p in PERSONS])
    r = paired(f1, f0)

    # 오라클 (테스트 라벨 전부로 offset 추정)
    off = np.array([(pred[person == p] - gaze[person == p]).mean(0) for p in PERSONS])
    oracle = angular_error(pred - off[[PERSONS.index(p) for p in person]], gaze).mean()

    print(f"보정 없음   {err.mean():.3f}°")
    print(f"라벨 0장    {err_z.mean():.3f}°   Δ {r['delta']:+.3f}° ± {r['se']:.3f}  "
          f"t={r['t']:.2f}  p={r['p']:.2e}  → {'유의' if r['significant'] else '구분 불가'}")
    print(f"오라클      {oracle:.3f}°  (테스트 라벨 3,000장)")
    print(f"회수율      {(err.mean() - err_z.mean()) / (err.mean() - oracle):.1%}")

    # 몇 장의 **라벨 없는** 이미지가 필요한가
    rng = np.random.default_rng(0)
    curve = {}
    for n in NS:
        vals = []
        for _ in range(BOOT):
            fe = []
            for p in PERSONS:
                m = np.flatnonzero(person == p)
                take = rng.choice(m, size=min(n, len(m)), replace=False)
                sh = pred[take].mean(0) - mu_train[p]
                fe.append(angular_error(pred[m] - sh, gaze[m]).mean())
            vals.append(float(np.mean(fe)))
        v = np.array(vals)
        curve[n] = (v.mean(), *np.percentile(v, [2.5, 97.5]))
        print(f"  라벨 없는 이미지 {n:>5}장 → {v.mean():.3f}° "
              f"[{curve[n][1]:.3f}, {curve[n][2]:.3f}]")

    print(f"\n좌우 불일치와의 관계 (구조적 검증)")
    s_yaw = None
    try:
        df = dataset.label_frame()
        eye = df["which_eye"].to_numpy()
        key = (df["person"].astype(str) + "|" + df["day"].astype(str) + "|"
               + df["frame"].astype(str)).to_numpy()
        o = np.argsort(key, kind="stable"); _, st = np.unique(key[o], return_index=True)
        li, ri = [], []
        for g in np.split(o, st[1:]):
            l, rr = g[eye[g] == "left"], g[eye[g] == "right"]
            if len(l) and len(rr):
                li.append(l[0]); ri.append(rr[0])
        li, ri = np.array(li), np.array(ri)
        dis = DEG(pred[li] - pred[ri] * np.array([-1.0, 1.0], dtype=np.float32))
        s_yaw = np.array([dis[person[li] == p][:, 0].mean() for p in PERSONS])
        slope, intercept = np.polyfit(s_yaw, DEG(off[:, 0]), 1)
        rr_, pp_ = pearson(s_yaw, DEG(off[:, 0]))
        print(f"  좌우 yaw 불일치 → offset yaw: 기울기 {slope:.3f} (구조상 기대 0.5), "
              f"r = {rr_:+.3f} (p = {pp_:.1e})")
        print(f"  이유: 같은 프레임의 yaw 라벨이 서로 반대라(gaze_L ≈ −gaze_R) "
              "두 예측을 더하면 라벨이 소거되고 치우침의 합만 남는다.")
        put_numbers(**{f"{tag}_a11_disagree_slope": round(float(slope), 4),
                       f"{tag}_a11_disagree_r": round(float(rr_), 4)})
    except Exception as exc:
        print(f"  (건너뜀: {exc})")

    put_numbers(**{
        f"{tag}_a11_zeroshot_error": round(float(err_z.mean()), 4),
        f"{tag}_a11_zeroshot_delta": round(float(r["delta"]), 4),
        f"{tag}_a11_zeroshot_se": round(float(r["se"]), 4),
        f"{tag}_a11_zeroshot_p": float(f"{r['p']:.3e}"),
        f"{tag}_a11_oracle_error": round(float(oracle), 4),
        f"{tag}_a11_recovery_pct": round(float((err.mean() - err_z.mean())
                                               / (err.mean() - oracle)), 4),
        f"{tag}_a11_n_curve": {str(n): round(curve[n][0], 4) for n in NS},
    })
    save_table(f"a11_zeroshot_{tag}", ["n_unlabeled_images", "mean_deg", "ci_lo", "ci_hi"],
               [[n, round(curve[n][0], 4), round(curve[n][1], 4), round(curve[n][2], 4)]
                for n in NS])
    save_table(f"a11_perfold_{tag}", ["person", "before_deg", "after_deg", "delta_deg",
                                      "shift_yaw_deg", "shift_pitch_deg"],
               [[p, round(float(f0[i]), 3), round(float(f1[i]), 3),
                 round(float(f1[i] - f0[i]), 3),
                 round(float(DEG(shifts[p][0])), 3), round(float(DEG(shifts[p][1])), 3)]
                for i, p in enumerate(PERSONS)])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    xs = np.array(NS)
    mu = np.array([curve[n][0] for n in NS])
    lo = np.array([curve[n][1] for n in NS]); hi = np.array([curve[n][2] for n in NS])
    ax.fill_between(xs, lo, hi, alpha=0.25, color="#1f4e79")
    ax.plot(xs, mu, "o-", ms=4, c="#1f4e79", label="라벨 0장 재중심화")
    ax.axhline(err.mean(), ls=":", lw=0.9, c="k", label=f"보정 없음 {err.mean():.2f}°")
    ax.axhline(oracle, ls="--", lw=0.9, c="#c0504d", label=f"오라클 {oracle:.2f}°")
    ax.set(xscale="log", xlabel="재중심화에 쓴 **라벨 없는** 이미지 수 (로그)",
           ylabel="LOPO 평균 각도 오차 (°)")
    ax.set_xticks(xs); ax.set_xticklabels([str(n) for n in NS])
    ax.legend(fontsize=6)

    ax = axes[1]
    w = 0.38; x = np.arange(len(PERSONS))
    ax.bar(x - w / 2, f0, w, label="보정 없음", color="#8aa0b8")
    ax.bar(x + w / 2, f1, w, label="라벨 0장", color="#1f4e79")
    ax.set_xticks(x, PERSONS, rotation=90); ax.set(ylabel="fold 오차 (°)")
    ax.legend(fontsize=6)

    ax = axes[2]
    if s_yaw is not None:
        ax.scatter(s_yaw, DEG(off[:, 0]), s=30, c="#1f4e79")
        for i, p in enumerate(PERSONS):
            ax.annotate(p, (s_yaw[i], DEG(off[i, 0])), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
        xs2 = np.linspace(s_yaw.min(), s_yaw.max(), 10)
        ax.plot(xs2, np.polyval(np.polyfit(s_yaw, DEG(off[:, 0]), 1), xs2),
                "--", lw=0.9, c="#1f4e79")
        ax.set(xlabel="좌우 예측 불일치 yaw (°) — 라벨 불필요",
               ylabel="실제 offset yaw (°)")
    save_fig(fig, "diag_23_zeroshot.png",
             {"person": PERSONS, "before": f0, "after": f1})

    print(f"\n결론: 라벨을 한 장도 쓰지 않고 오차를 {err.mean():.2f}° → {err_z.mean():.2f}°로 "
          f"줄인다(p={r['p']:.1e}). 오라클 회수율 {(err.mean()-err_z.mean())/(err.mean()-oracle):.0%}. "
          "다만 테스트 사람의 이미지 묶음으로 통계를 내는 transductive 방법이라 "
          "과제 규칙상 메인 수치에는 넣지 않는다.")


if __name__ == "__main__":
    run()
