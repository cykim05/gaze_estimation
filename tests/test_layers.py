"""계층 수치 기울기 검증. `python -m tests.test_layers`

역전파를 직접 쓴 코드라 해석적 기울기를 수치 미분과 맞춰 보지 않으면 믿을 수 없다.
float32로는 수치 미분 오차가 커서 검증에서만 float64로 올려 잰다.
"""

import numpy as np

from common import layers
from common.layers import (
    Affine,
    BatchNormalization,
    Convolution,
    Dropout,
    GradientReversal,
    GroupNormalization,
    MeanSquaredError,
    Pooling,
    Relu,
    SoftmaxWithLoss,
    WeightedMeanSquaredError,
)

RNG = np.random.default_rng(0)


def numerical_grad(f, x, h=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + h
        f1 = f()
        x[i] = orig - h
        f2 = f()
        x[i] = orig
        grad[i] = (f1 - f2) / (2 * h)
        it.iternext()
    return grad


def rel_err(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def check(name, layer, x, params=(), train_flg=None, tol=1e-5):
    """스칼라 손실 sum(weight * out)에 대해 dx와 파라미터 기울기를 수치 미분과 비교한다."""

    def forward():
        return layer.forward(x) if train_flg is None else layer.forward(x, train_flg)

    def run():
        return float(np.sum(forward() * weight))

    weight = RNG.standard_normal(forward().shape)

    loss = run()
    dx = layer.backward(weight.copy())

    dx_num = numerical_grad(run, x)
    errs = [("dx", rel_err(dx, dx_num))]
    for pname in params:
        p = getattr(layer, pname)
        g_num = numerical_grad(run, p)
        g = getattr(layer, "d" + pname)
        errs.append(("d" + pname, rel_err(g, g_num)))

    report = "  ".join(f"{k}={v:.2e}" for k, v in errs)
    worst = max(v for _, v in errs)
    status = "ok " if worst < tol else "FAIL"
    print(f"[{status}] {name:<22} loss={loss:+.4f}  {report}")
    assert worst < tol, f"{name}: 기울기 불일치 {report}"


def main():
    layers.DTYPE = np.float64  # 정규화 계층 파라미터 생성에 쓰인다

    print("=== 계층별 수치 기울기 검증 (float64) ===")

    check("Relu", Relu(), RNG.standard_normal((4, 6)))

    x = RNG.standard_normal((4, 6))
    affine = Affine(RNG.standard_normal((6, 3)), RNG.standard_normal(3))
    check("Affine", affine, x, params=("W", "b"))

    x4 = RNG.standard_normal((3, 2, 7, 9))
    conv = Convolution(RNG.standard_normal((4, 2, 3, 3)), RNG.standard_normal(4), stride=1, pad=1)
    check("Convolution pad1", conv, x4, params=("W", "b"))

    conv2 = Convolution(RNG.standard_normal((3, 2, 5, 5)), RNG.standard_normal(3), stride=2, pad=0)
    check("Convolution s2 5x5", conv2, x4, params=("W", "b"))

    check("Pooling 2x2", Pooling(2, 2), RNG.standard_normal((3, 2, 6, 8)))
    check("Pooling 3x3 s2", Pooling(3, 3, stride=2), RNG.standard_normal((2, 3, 7, 9)))

    bn2 = BatchNormalization(6)
    bn2.gamma = RNG.standard_normal(6)
    bn2.beta = RNG.standard_normal(6)
    check("BatchNorm 2D", bn2, RNG.standard_normal((8, 6)), params=("gamma", "beta"), train_flg=True)

    bn4 = BatchNormalization(2)
    bn4.gamma = RNG.standard_normal(2)
    bn4.beta = RNG.standard_normal(2)
    check("BatchNorm 4D", bn4, RNG.standard_normal((5, 2, 3, 4)),
          params=("gamma", "beta"), train_flg=True, tol=1e-4)

    gn4 = GroupNormalization(6, groups=3)
    gn4.gamma = RNG.standard_normal(6)
    gn4.beta = RNG.standard_normal(6)
    check("GroupNorm 4D", gn4, RNG.standard_normal((4, 6, 5, 7)),
          params=("gamma", "beta"), train_flg=True, tol=1e-4)

    # 그룹당 원소가 2개뿐이면 std가 |a-b|/2라 xn이 ±1로 고정되고 수치 미분이
    # 무너진다. 모델이 실제로 쓰는 크기(fc 256채널을 8그룹 = 그룹당 32개)에 맞춘다.
    gn2 = GroupNormalization(16, groups=2)
    gn2.gamma = RNG.standard_normal(16)
    gn2.beta = RNG.standard_normal(16)
    check("GroupNorm 2D", gn2, RNG.standard_normal((5, 16)),
          params=("gamma", "beta"), train_flg=True, tol=1e-4)

    # GradientReversal: 순전파 항등, 역전파 부호 반전
    grl = GradientReversal(scale=0.7)
    v = RNG.standard_normal((3, 4))
    g = RNG.standard_normal((3, 4))
    ok = np.array_equal(grl.forward(v), v) and np.allclose(grl.backward(g), -0.7 * g)
    print(f"[{'ok ' if ok else 'FAIL'}] {'GradientReversal':<22} 순전파 항등, 역전파 ×(-0.7)")
    assert ok

    # SoftmaxWithLoss: 교차 엔트로피 기울기
    logits = RNG.standard_normal((6, 4))
    labels = RNG.integers(0, 4, 6)
    sm = SoftmaxWithLoss()
    sm.forward(logits, labels)
    g_an = sm.backward()
    g_num = numerical_grad(lambda: sm.forward(logits, labels), logits)
    e = rel_err(g_an, g_num)
    print(f"[{'ok ' if e < 1e-6 else 'FAIL'}] {'SoftmaxWithLoss':<22} dx={e:.2e}")
    assert e < 1e-6

    # WeightedMeanSquaredError: 가중치가 있을 때와 없을 때 모두
    yv, tv = RNG.standard_normal((5, 2)), RNG.standard_normal((5, 2))
    wv = np.abs(RNG.standard_normal(5)); wv /= wv.sum()
    for name, w in (("가중치 없음", None), ("가중치 있음", wv)):
        wm = WeightedMeanSquaredError()
        wm.forward(yv, tv, w)
        g_an = wm.backward()
        g_num = numerical_grad(lambda: wm.forward(yv, tv, w), yv)
        e = rel_err(g_an, g_num)
        print(f"[{'ok ' if e < 1e-6 else 'FAIL'}] {'WeightedMSE(' + name + ')':<22} dy={e:.2e}")
        assert e < 1e-6

    # MeanSquaredError: 손실이 스칼라라 위 틀과 모양이 달라 따로 본다
    y = RNG.standard_normal((5, 2))
    t = RNG.standard_normal((5, 2))
    mse = MeanSquaredError()
    g = mse.backward(1.0) if mse.forward(y, t) is not None else None
    g_num = numerical_grad(lambda: mse.forward(y, t), y)
    print(f"[{'ok ' if rel_err(g, g_num) < 1e-6 else 'FAIL'}] {'MeanSquaredError':<22} "
          f"loss={mse.forward(y, t):+.4f}  dy={rel_err(g, g_num):.2e}")
    assert rel_err(g, g_num) < 1e-6

    # Dropout: 추론 시 항등, 학습 시 기댓값 보존
    # 평균이 0인 입력으로는 비율을 볼 수 없어 1로 채운 배열을 쓴다
    drop = Dropout(0.3)
    ones = np.ones((2000, 50))
    assert np.array_equal(drop.forward(ones, train_flg=False), ones), "추론 시 항등이 아니다"
    scale = drop.forward(ones, train_flg=True).mean()
    print(f"[{'ok ' if abs(scale - 1.0) < 0.02 else 'FAIL'}] {'Dropout(0.3)':<22} "
          f"E[out]=1 유지: {scale:.4f}")
    assert abs(scale - 1.0) < 0.02

    print("\n전부 통과.")


if __name__ == "__main__":
    main()
