"""A13 — 남은 오차 중 얼마가 줄일 수 있는 것인가. **[진단용]** (자율 탐색)

같은 프레임의 두 눈은 **같은 장면의 두 번의 독립 측정**에 가깝다. 두 잔차의 상관이
공유 성분(장면·사람 수준, 더 많은 눈 이미지를 평균해도 안 줄어듦)의 비중을 알려 준다.

    Var(잔차) = Var(공유) + Var(독립)
    corr(두 눈) ≈ Var(공유) / Var(잔차)

공유 성분을 다시 '사람 고정'과 '사람 안 장면'으로 가른다.
"""
from __future__ import annotations

import numpy as np

from analysis.common import (DEG, PERSONS, angular_error, load_run, pearson, plt,
                             put_numbers, save_fig, save_table, section)

FLIP = np.array([-1.0, 1.0], dtype=np.float32)


def run(run_dir="results/final", tag="runB"):
    section(f"A13 — 오차의 줄일 수 있는 부분 [진단용] ({run_dir})")
    import dataset
    d = load_run(run_dir)
    df = dataset.label_frame()
    person, pred, gaze = d["person"], d["pred"], d["gaze"]
    eye = df["which_eye"].to_numpy()
    key = (df["person"].astype(str) + "|" + df["day"].astype(str) + "|"
           + df["frame"].astype(str)).to_numpy()
    o = np.argsort(key, kind="stable"); _, st = np.unique(key[o], return_index=True)
    li, ri = [], []
    for g in np.split(o, st[1:]):
        l, r = g[eye[g] == "left"], g[eye[g] == "right"]
        if len(l) and len(r):
            li.append(l[0]); ri.append(r[0])
    li, ri = np.array(li), np.array(ri)

    res_l = DEG(pred[li] - gaze[li])
    res_r = DEG(pred[ri] * FLIP - gaze[ri] * FLIP)
    pl = person[li]

    print(f"쌍 {len(li):,}개 · 세 층으로 분해 (단위 도²)\n")
    rows = []
    for k, axis in enumerate(("yaw", "pitch")):
        a, b = res_l[:, k], res_r[:, k]
        total = np.var(np.concatenate([a, b]), ddof=1)
        rho, _ = pearson(a, b)
        shared = max(rho, 0.0) * total          # 두 눈이 공유하는 성분
        indep = total - shared                   # 눈마다 독립인 성분

        # 공유 성분 안에서 사람 고정 성분
        pair_mean = (a + b) / 2
        per_person = np.array([pair_mean[pl == p].mean() for p in PERSONS])
        var_person = per_person.var(ddof=1)
        var_scene = max(shared - var_person, 0.0)

        print(f"{axis}")
        print(f"  전체 잔차 분산            {total:8.3f}")
        print(f"  ├ 두 눈 공유 ({rho:+.3f})     {shared:8.3f}  ({shared/total:5.1%})")
        print(f"  │  ├ 사람 고정 (offset)   {var_person:8.3f}  ({var_person/total:5.1%})")
        print(f"  │  └ 장면 (자세·조명)      {var_scene:8.3f}  ({var_scene/total:5.1%})")
        print(f"  └ 눈마다 독립 (이미지)     {indep:8.3f}  ({indep/total:5.1%})")
        rows.append([axis, round(float(total), 4), round(float(rho), 4),
                     round(float(shared), 4), round(float(var_person), 4),
                     round(float(var_scene), 4), round(float(indep), 4),
                     round(float(shared / total), 4), round(float(indep / total), 4)])
        put_numbers(**{
            f"{tag}_a13_{axis}_total_var": round(float(total), 4),
            f"{tag}_a13_{axis}_shared_frac": round(float(shared / total), 4),
            f"{tag}_a13_{axis}_person_frac": round(float(var_person / total), 4),
            f"{tag}_a13_{axis}_scene_frac": round(float(var_scene / total), 4),
            f"{tag}_a13_{axis}_indep_frac": round(float(indep / total), 4),
        })
    save_table(f"a13_variance_layers_{tag}",
               ["axis", "total_var", "two_eye_corr", "shared_var", "person_var",
                "scene_var", "independent_var", "shared_frac", "independent_frac"], rows)

    # 실제로 눈을 더 평균하면 어디까지 가나
    print(f"\n같은 프레임의 눈을 평균했을 때 (실측)")
    one = (angular_error(pred[li], gaze[li]).mean()
           + angular_error(pred[ri] * FLIP, gaze[ri] * FLIP).mean()) / 2
    two = angular_error((pred[li] + pred[ri] * FLIP) / 2, gaze[li]).mean()
    print(f"  눈 1개 {one:.3f}°  →  눈 2개 평균 {two:.3f}°  ({two - one:+.3f}°)")
    print(f"  독립 성분이 {rows[1][8]:.0%}(pitch)라면 2개 평균의 이론적 하한은 "
          "여기서 더 내려가기 어렵다.")
    put_numbers(**{f"{tag}_a13_one_eye": round(float(one), 4),
                   f"{tag}_a13_two_eye": round(float(two), 4)})

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    labels = ["yaw", "pitch"]
    person_f = [r[4] / r[1] for r in rows]
    scene_f = [r[5] / r[1] for r in rows]
    indep_f = [r[6] / r[1] for r in rows]
    x = np.arange(2)
    ax.bar(x, person_f, 0.5, label="사람 고정 (offset)", color="#1f4e79")
    ax.bar(x, scene_f, 0.5, bottom=person_f, label="장면 (자세·조명)", color="#4f81bd")
    ax.bar(x, indep_f, 0.5, bottom=np.array(person_f) + np.array(scene_f),
           label="눈마다 독립 (이미지)", color="#c6d9f0")
    ax.set_xticks(x, labels); ax.set(ylabel="잔차 분산 중 비중")
    ax.legend(fontsize=6)
    save_fig(fig, "diag_26_variance_layers.png",
             {"axis": labels, "person": person_f, "scene": scene_f, "independent": indep_f})


if __name__ == "__main__":
    run()
