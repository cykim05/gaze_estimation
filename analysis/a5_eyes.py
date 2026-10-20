"""A5 — 좌우 눈 일관성. **[진단용]** (`WhichEye`, `Origin` 사용)

질문: 같은 프레임에서 나온 좌·우눈의 오차가 상관되어 있나? 두 눈을 합치면 이득인가?

**좌표 정렬이 먼저다.** 전처리가 오른눈을 좌우 반전해 왼눈처럼 저장하므로,
같은 프레임의 라벨은 yaw_L ≈ −yaw_R, pitch_L ≈ +pitch_R 이다(실측 상관 ∓0.999).
정렬하지 않고 상관을 재면 yaw가 음수로 나와 해석이 뒤집힌다. 모든 양을
'왼눈 기준'으로 (yaw → −yaw) 맞춘 뒤 비교한다.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, angular_error, load_run, pearson, plt,
                             put_numbers, save_fig, save_table, section)

FLIP = np.array([-1.0, 1.0], dtype=np.float32)   # yaw만 뒤집는다


def run(run_dir="results/final", tag="runB"):
    section(f"A5 — 좌우 눈 일관성 [진단용] ({run_dir})")
    import dataset
    d = load_run(run_dir)
    df = dataset.label_frame()
    assert (df["person"].to_numpy() == d["person"]).all(), "사람 순서가 다르다"

    eye = df["which_eye"].to_numpy()
    key = (df["person"].astype(str) + "|" + df["day"].astype(str) + "|"
           + df["frame"].astype(str)).to_numpy()
    order = np.argsort(key, kind="stable")
    _, starts = np.unique(key[order], return_index=True)
    li, ri = [], []
    for g in np.split(order, starts[1:]):
        l, r = g[eye[g] == "left"], g[eye[g] == "right"]
        if len(l) and len(r):
            li.append(l[0]); ri.append(r[0])
    li, ri = np.array(li), np.array(ri)

    # 왼눈 기준으로 정렬한 예측·정답
    pred_l, gaze_l = d["pred"][li], d["gaze"][li]
    pred_r, gaze_r = d["pred"][ri] * FLIP, d["gaze"][ri] * FLIP
    print(f"좌·우눈이 모두 있는 프레임 {len(li):,}쌍 ({2 * len(li) / len(df):.0%}의 샘플)")
    print(f"  정렬 확인: |gaze_L − gaze_R| 평균 "
          f"{DEG(np.abs(gaze_l - gaze_r)).mean(0).round(2)} 도 (0에 가까워야 한다)")

    res_l, res_r = DEG(pred_l - gaze_l), DEG(pred_r - gaze_r)
    err_l = angular_error(pred_l, gaze_l)
    err_r = angular_error(pred_r, gaze_r)

    r_err, p_err = pearson(err_l, err_r)
    r_yaw, p_yaw = pearson(res_l[:, 0], res_r[:, 0])
    r_pit, p_pit = pearson(res_l[:, 1], res_r[:, 1])
    print(f"\n같은 프레임 (정렬 후)")
    print(f"  오차 크기 상관   r = {r_err:+.3f}  (p = {p_err:.1e})")
    print(f"  yaw 잔차 상관    r = {r_yaw:+.3f}  (p = {p_yaw:.1e})")
    print(f"  pitch 잔차 상관  r = {r_pit:+.3f}  (p = {p_pit:.1e})")

    # 비교 기준: 같은 사람·같은 눈종류, 다른 장면
    rng = np.random.default_rng(0)
    ra, rb = [], []
    for p in PERSONS:
        for which in ("left", "right"):
            idx = np.flatnonzero((d["person"] == p) & (eye == which))
            a, b = rng.permutation(idx), rng.permutation(idx)
            m = a != b
            ra.append(a[m]); rb.append(b[m])
    ra, rb = np.concatenate(ra), np.concatenate(rb)
    res_all = DEG(d["pred"] - d["gaze"])
    r_rand_y, _ = pearson(res_all[ra, 0], res_all[rb, 0])
    r_rand_p, _ = pearson(res_all[ra, 1], res_all[rb, 1])
    r_rand_e, _ = pearson(d["err"][ra], d["err"][rb])
    print(f"\n같은 사람·같은 눈·다른 장면 (무작위 쌍 {len(ra):,}개)")
    print(f"  오차 r = {r_rand_e:+.3f}   yaw 잔차 r = {r_rand_y:+.3f}   "
          f"pitch 잔차 r = {r_rand_p:+.3f}")

    # 두 눈 예측 평균 (정렬 후) — 정당한 결합
    comb = (pred_l + pred_r) / 2
    err_comb = angular_error(comb, gaze_l)
    solo = (err_l + err_r) / 2
    print(f"\n두 눈 결합 (왼눈 기준으로 정렬해 평균)")
    print(f"  개별 평균 {solo.mean():.3f}°  →  결합 {err_comb.mean():.3f}°  "
          f"({err_comb.mean() - solo.mean():+.3f}°)")
    print(f"  왼눈 {err_l.mean():.3f}°   오른눈(정렬) {err_r.mean():.3f}°")

    put_numbers(**{
        f"{tag}_eye_pairs": int(len(li)),
        f"{tag}_eye_err_corr_aligned": round(r_err, 4),
        f"{tag}_eye_yaw_resid_corr_aligned": round(r_yaw, 4),
        f"{tag}_eye_pitch_resid_corr_aligned": round(r_pit, 4),
        f"{tag}_eye_random_err_corr": round(r_rand_e, 4),
        f"{tag}_eye_random_yaw_corr": round(r_rand_y, 4),
        f"{tag}_eye_random_pitch_corr": round(r_rand_p, 4),
        f"{tag}_eye_left_err": round(float(err_l.mean()), 4),
        f"{tag}_eye_right_err_aligned": round(float(err_r.mean()), 4),
        f"{tag}_eye_combined_err": round(float(err_comb.mean()), 4),
        f"{tag}_eye_combine_gain": round(float(solo.mean() - err_comb.mean()), 4),
    })
    save_table(f"a5_eye_consistency_{tag}", ["quantity", "value"], [
        ["pairs", int(len(li))],
        ["err_corr_same_frame_aligned", round(r_err, 4)],
        ["yaw_resid_corr_same_frame_aligned", round(r_yaw, 4)],
        ["pitch_resid_corr_same_frame_aligned", round(r_pit, 4)],
        ["err_corr_random_same_person_same_eye", round(r_rand_e, 4)],
        ["yaw_resid_corr_random", round(r_rand_y, 4)],
        ["pitch_resid_corr_random", round(r_rand_p, 4)],
        ["left_eye_err_deg", round(float(err_l.mean()), 4)],
        ["right_eye_err_deg_aligned", round(float(err_r.mean()), 4)],
        ["two_eye_combined_err_deg", round(float(err_comb.mean()), 4)],
        ["combine_gain_deg", round(float(solo.mean() - err_comb.mean()), 4)]])

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2))
    axes[0].hexbin(res_l[:, 1], res_r[:, 1], gridsize=45, cmap="Greys", bins="log", mincnt=1)
    axes[0].set(xlabel="왼눈 pitch 잔차 (°)", ylabel="오른눈 pitch 잔차 (°, 정렬)")
    axes[0].text(0.03, 0.95, f"같은 프레임 r = {r_pit:+.3f}",
                 transform=axes[0].transAxes, fontsize=7, va="top")
    axes[1].hexbin(res_all[ra, 1], res_all[rb, 1], gridsize=45, cmap="Greys",
                   bins="log", mincnt=1)
    axes[1].set(xlabel="무작위 쌍 A pitch 잔차 (°)", ylabel="무작위 쌍 B pitch 잔차 (°)")
    axes[1].text(0.03, 0.95, f"같은 사람·다른 장면 r = {r_rand_p:+.3f}",
                 transform=axes[1].transAxes, fontsize=7, va="top")
    x = np.arange(3); w = 0.38
    axes[2].bar(x - w / 2, [r_yaw, r_pit, r_err], w, label="같은 프레임", color="#1f4e79")
    axes[2].bar(x + w / 2, [r_rand_y, r_rand_p, r_rand_e], w,
                label="같은 사람·다른 장면", color="#8aa0b8")
    axes[2].set_xticks(x, ["yaw 잔차", "pitch 잔차", "오차 크기"])
    axes[2].set(ylabel="상관계수 r")
    axes[2].legend(fontsize=6)
    save_fig(fig, "diag_17_eye_consistency.png",
             {"left_pitch_resid": res_l[:, 1], "right_pitch_resid_aligned": res_r[:, 1]})

    print(f"\n결론: 같은 프레임의 두 눈은 pitch 잔차가 r={r_pit:+.2f}로 거의 같이 움직인다. "
          f"같은 사람이라도 장면이 다르면 {r_rand_p:+.2f}로 떨어진다. "
          "오차의 큰 부분이 이미지 잡음이 아니라 **장면(자세·조명) 수준**에서 생긴다.")


if __name__ == "__main__":
    run()
