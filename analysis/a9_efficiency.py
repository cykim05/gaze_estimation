"""A9 — 효율: 처리량 격자와 정확도-비용 Pareto.

질문: 같은 GPU 한 장으로 얼마나 더 많은 실험을 돌릴 수 있나? 레버는 배치인가 패킹인가?
PREREG 1절의 사전 예상은 "per-gpu는 2~3에서 포화, 배치가 더 큰 레버"였다.
"""
from __future__ import annotations

import json
import os

import numpy as np

from analysis.common import (plt, put_numbers, save_fig, save_table, section)

B0 = "results/bench/b0_throughput.json"


def run():
    section("A9 — 처리량과 정확도-비용")
    rows = json.load(open(B0, encoding="utf-8"))
    ok = [r for r in rows if "samples_per_s" in r]
    batches = sorted({r["batch"] for r in ok})
    pergpus = sorted({r["per_gpu"] for r in ok})

    grid = {(r["batch"], r["per_gpu"]): r for r in ok}
    print(f"{'batch':>7}" + "".join(f"{'pg=' + str(k):>12}" for k in pergpus)
          + f"{'최고':>10}{'최고 pg':>9}")
    table, best_overall = [], (None, -1)
    for b in batches:
        line = f"{b:>7}"
        rates = []
        for k in pergpus:
            r = grid.get((b, k))
            rates.append(r["samples_per_s"] if r else np.nan)
            line += f"{r['samples_per_s']:>12,.0f}" if r else f"{'OOM':>12}"
        rates = np.array(rates, float)
        bi = int(np.nanargmax(rates))
        line += f"{rates[bi]:>10,.0f}{pergpus[bi]:>9}"
        print(line)
        if rates[bi] > best_overall[1]:
            best_overall = ((b, pergpus[bi]), rates[bi])
        table.append([b] + [None if np.isnan(v) else round(float(v)) for v in rates]
                     + [round(float(rates[bi])), pergpus[bi]])

    save_table("a9_throughput_grid",
               ["batch"] + [f"samples_per_s_pergpu{k}" for k in pergpus]
               + ["best_samples_per_s", "best_per_gpu"], table)

    # 패킹 이득 (batch별)
    print(f"\n패킹 이득 (per-gpu 1 대비)")
    gains = {}
    for b in batches:
        base = grid.get((b, 1))
        if not base:
            continue
        g = {k: (grid[(b, k)]["samples_per_s"] / base["samples_per_s"]
                 if (b, k) in grid else np.nan) for k in pergpus}
        gains[b] = g
        print(f"  batch {b:>4}: " + "  ".join(
            f"pg={k} {g[k]:.2f}×" if not np.isnan(g[k]) else f"pg={k} OOM" for k in pergpus)
            + f"   (util pg=1 {base['util_pct']:.0f}%)")

    single = {b: grid[(b, 1)]["samples_per_s"] for b in batches if (b, 1) in grid}
    put_numbers(
        b0_best_config=f"batch {best_overall[0][0]}, per-gpu {best_overall[0][1]}",
        b0_best_samples_per_s=round(float(best_overall[1])),
        b0_pack_gain_batch128_pg2=round(float(gains[128][2]), 3),
        b0_pack_gain_batch512_pg2=round(float(gains[512][2]), 3),
        b0_single_proc_rate_b128=round(float(single[128])),
        b0_single_proc_rate_b1024=round(float(single[1024])),
        b0_util_pct_b128_pg1=round(float(grid[(128, 1)]["util_pct"]), 1),
        b0_util_pct_b1024_pg1=round(float(grid[(1024, 1)]["util_pct"]), 1),
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    ax = axes[0]
    for k in pergpus:
        ys = [grid[(b, k)]["samples_per_s"] if (b, k) in grid else np.nan for b in batches]
        ax.plot(batches, ys, "o-", ms=4, label=f"per-gpu {k}")
    ax.set(xscale="log", xlabel="배치 크기 (로그 눈금)", ylabel="합계 처리량 (samples/s)")
    ax.set_xticks(batches); ax.set_xticklabels([str(b) for b in batches])
    ax.legend(fontsize=6)

    ax = axes[1]
    for k in pergpus:
        ys = [gains[b][k] if b in gains else np.nan for b in batches]
        ax.plot(batches, ys, "o-", ms=4, label=f"per-gpu {k}")
    ax.axhline(1.0, ls="--", lw=0.8, c="k")
    ax.axhline(1.1, ls=":", lw=0.8, c="#c0504d")
    ax.set(xscale="log", xlabel="배치 크기 (로그 눈금)",
           ylabel="per-gpu 1 대비 처리량 배수")
    ax.set_xticks(batches); ax.set_xticklabels([str(b) for b in batches])
    ax.legend(fontsize=6)
    ax.text(0.03, 0.05, "점선: PREREG의 10% 채택 기준", transform=ax.transAxes, fontsize=6)

    ax = axes[2]
    for k in pergpus:
        xs = [grid[(b, k)]["mem_mib"] if (b, k) in grid else np.nan for b in batches]
        ys = [grid[(b, k)]["util_pct"] if (b, k) in grid else np.nan for b in batches]
        ax.plot(xs, ys, "o-", ms=4, label=f"per-gpu {k}")
    ax.set(xlabel="GPU 메모리 사용 (MiB)", ylabel="GPU 사용률 (%)")
    ax.legend(fontsize=6)
    save_fig(fig, "15_throughput.png",
             {"batch": [r["batch"] for r in ok], "per_gpu": [r["per_gpu"] for r in ok],
              "samples_per_s": [r["samples_per_s"] for r in ok],
              "util_pct": [r["util_pct"] for r in ok],
              "mem_mib": [r["mem_mib"] for r in ok]})

    print(f"\n사전 예상과의 대조 (PREREG 1절)")
    print(f"  예상: per-gpu는 2~3에서 포화, **배치가 더 큰 레버**")
    print(f"  실제: 배치는 레버가 아니다. 단일 프로세스 처리량이 "
          f"{single[128]:,.0f} → {single[1024]:,.0f} samples/s로 오히려 "
          f"{1 - single[1024]/single[128]:.0%} 줄었다.")
    print(f"        패킹은 batch 128에서만 이득({gains[128][2]:.2f}×). "
          f"256 이상에서는 1.0 근처다({gains[256][2]:.2f}, {gains[512][2]:.2f}).")
    print(f"  해석: batch 128은 단일 프로세스가 카드를 못 채운다(util "
          f"{grid[(128,1)]['util_pct']:.0f}%). 두 프로세스가 빈틈을 메운다. "
          f"배치를 키우면 한 프로세스가 이미 채우므로(util "
          f"{grid[(1024,1)]['util_pct']:.0f}%) 패킹이 시분할일 뿐이다.")


if __name__ == "__main__":
    run()
