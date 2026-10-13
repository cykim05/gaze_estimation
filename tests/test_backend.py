"""CPU와 GPU가 같은 계산을 하는지 확인한다. `python -m tests.test_backend`

`common/backend.py`는 import 시점에 배열 모듈을 고르므로 한 프로세스 안에서 둘을
같이 쓸 수 없다. 자식 프로세스를 둘 띄워 각자 결과를 npz로 내리고 부모가 비교한다.

**두 단계로 본다.**

1. float64 — 알고리즘이 같은가. 누적 오차가 묻히는 정밀도에서 재므로, 여기서
   어긋나면 구현이 다른 것이다. 기준 1e-10.
2. float32 — 실제로 쓰는 정밀도. 순전파와 손실이 맞는지만 판정한다.
   기울기는 숫자를 찍되 판정에 넣지 않는다. 평균이 0에 가까운 값을 수만 개 더하는
   축(BN의 dbeta는 8,640개, conv의 dW는 138,240개)에서는 상쇄가 일어나, 더하는
   순서만 달라도 **상대** 오차가 1e-1까지 간다. 구현이 아니라 float32의 성질이다.
   구현이 같다는 증거는 1번이 댄다.

학습을 몇 스텝 돌린 뒤의 파라미터는 비교하지 않는다. Adam의 보폭은 첫 스텝에서
`lr * g / (|g| + eps)`라 기울기 부호만 뒤집혀도 결과가 lr만큼 갈린다. 기울기가 0
근처인 성분에서는 반드시 갈리므로, 비교해도 의미가 없다. 대신 손실이 붙어 있는지만 본다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

BATCH = 64
TOL = {"float64": 1e-10, "float32": 1e-5}        # 순전파·손실 기준
GRAD_TOL = {"float64": 1e-9}                     # float32는 판정하지 않는다 (docstring)
LOSS5_TOL = {"float64": 1e-8, "float32": 1e-2}   # 5스텝 뒤 손실 기준


def compute(path: str, dtype_name: str) -> None:
    """현재 장치·정밀도로 한 스텝을 돌리고 결과를 저장한다."""
    import common.layers
    import model
    from common.backend import describe, to_cpu, xp
    from common.optimizer import Adam
    from model import EYE_SHAPE, GazeNet

    dtype = getattr(np, dtype_name)
    common.layers.DTYPE = model.DTYPE = dtype
    print(f"  {describe()}  dtype={dtype_name}", flush=True)

    rng = np.random.default_rng(0)
    make = lambda *shape: xp.asarray(rng.standard_normal(shape).astype(dtype))
    image, head, gaze = make(BATCH, *EYE_SHAPE), make(BATCH, 2), make(BATCH, 2)

    net = GazeNet(conv_channels=(8, 16, 32), hidden=64, joint=32, dropout=0.0, seed=0)
    out = {
        "pred0": to_cpu(net.predict(image, head, train_flg=False)),
        "loss0": np.asarray(net.loss(image, head, gaze, train_flg=True), dtype=np.float64),
    }
    out.update({f"grad.{k}": to_cpu(v) for k, v in net.gradient(image, head, gaze).items()})

    opt = Adam(lr=1e-3)
    for _ in range(5):
        opt.update(net.params, net.gradient(image, head, gaze))
    out["loss5"] = np.asarray(net.loss(image, head, gaze, train_flg=False), dtype=np.float64)

    np.savez(path, **out)


def rel_err(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
    return float(np.abs(a - b).max() / scale)


def compare(dtype_name: str, tmp: str) -> bool:
    paths = {}
    for device in ("cpu", "gpu"):
        paths[device] = os.path.join(tmp, f"{device}-{dtype_name}.npz")
        subprocess.run([sys.executable, "-m", "tests.test_backend", "--compute",
                        paths[device], dtype_name],
                       env=dict(os.environ, GAZE_DEVICE=device), check=True)

    with np.load(paths["cpu"]) as a, np.load(paths["gpu"]) as b:
        forward = {k: rel_err(a[k], b[k]) for k in ("pred0", "loss0")}
        grads = {k: rel_err(a[k], b[k]) for k in a.files if k.startswith("grad.")}
        loss5 = rel_err(a["loss5"], b["loss5"])

    grad_tol = GRAD_TOL.get(dtype_name)
    ok = (max(forward.values()) < TOL[dtype_name]
          and loss5 < LOSS5_TOL[dtype_name]
          and (grad_tol is None or max(grads.values()) < grad_tol))

    worst_grad = max(grads, key=grads.get)
    print(f"\n  순전파  pred0 {forward['pred0']:.2e}  loss0 {forward['loss0']:.2e}"
          f"   (기준 {TOL[dtype_name]:.0e})")
    note = f"기준 {grad_tol:.0e}" if grad_tol else "판정 제외 — 상쇄 누적"
    print(f"  기울기  최악 {grads[worst_grad]:.2e} ({worst_grad[5:]}), "
          f"배열 {len(grads)}개   ({note})")
    print(f"  5스텝 뒤 손실  {loss5:.2e}   (기준 {LOSS5_TOL[dtype_name]:.0e})")
    print(f"  → {'일치' if ok else '불일치'}")
    return ok


def main() -> int:
    try:
        import cupy  # noqa: F401
    except ImportError:
        print("cupy가 없다. GPU 비교를 건너뛴다.")
        return 0

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for dtype_name in ("float64", "float32"):
            print(f"=== {dtype_name} ===")
            results[dtype_name] = compare(dtype_name, tmp)
            print()

    if all(results.values()):
        print("CPU와 GPU가 같은 계산을 한다.")
        print("float64에서 1e-10 안쪽으로 붙으므로 구현은 동일하고, float32에서 벌어지는"
              " 기울기 차이는\n더하는 순서가 달라 생기는 상쇄 오차다.")
        return 0
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--compute":
        compute(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
