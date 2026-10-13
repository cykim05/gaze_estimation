"""GazeNet 전체 역전파 검증과 처리량 측정. `python -m tests.test_model`

계층 하나하나가 맞아도 이어 붙이는 과정(특징 ⊕ head pose, 기울기 분리)에서
틀릴 수 있다. 손실에서 모든 파라미터까지의 기울기를 수치 미분과 대조한다.
"""

import time

import numpy as np

import common.layers
import model
from model import EYE_SHAPE, GazeNet

RNG = np.random.default_rng(0)


def grad_err(analytic, numeric, scale):
    """상대 오차. 단 기울기가 scale에 비해 무시할 만하면 절대 오차로 본다.

    BatchNormalization 바로 앞의 bias는 BN이 평균을 빼므로 출력에 아무 영향이 없다.
    참 기울기가 0이라 상대 오차를 재면 부동소수점 잡음을 재는 꼴이 된다.
    """
    denom = np.maximum(np.abs(analytic) + np.abs(numeric), scale * 1e-4)
    return float(np.max(np.abs(analytic - numeric) / denom))


def test_gradient():
    print("=== GazeNet 전체 기울기 검증 (float64, dropout 끔) ===")
    common.layers.DTYPE = model.DTYPE = np.float64

    net = GazeNet(conv_channels=(4, 6), hidden=12, joint=8, dropout=0.0, seed=0)
    n = 6
    image = RNG.standard_normal((n, *EYE_SHAPE))
    head = RNG.standard_normal((n, 2))
    gaze = RNG.standard_normal((n, 2))

    grads = net.gradient(image, head, gaze)
    scale = max(np.abs(g).max() for g in grads.values())

    worst = 0.0
    for key, param in net.params.items():
        # 파라미터가 큰 층은 몇 개만 찍어 본다(수치 미분은 원소마다 순전파 2회다)
        flat = param.ravel()
        picks = RNG.choice(flat.size, size=min(6, flat.size), replace=False)
        num = np.empty(len(picks))
        for j, i in enumerate(picks):
            orig = flat[i]
            flat[i] = orig + 1e-6
            f1 = net.loss(image, head, gaze, train_flg=True)
            flat[i] = orig - 1e-6
            f2 = net.loss(image, head, gaze, train_flg=True)
            flat[i] = orig
            num[j] = (f1 - f2) / 2e-6
        analytic = grads[key].ravel()[picks]
        err = grad_err(analytic, num, scale)
        worst = max(worst, err)
        dead = " (BN 앞 bias — 참 기울기 0)" if np.abs(analytic).max() < scale * 1e-9 else ""
        print(f"  {key:<12} {str(param.shape):<16} err {err:.2e}  "
              f"|grad|max {np.abs(analytic).max():.2e}{dead}")

    print(f"\n최악 {worst:.2e}")
    assert worst < 1e-5, "모델 역전파가 수치 미분과 다르다"


def test_head_pose_is_used():
    """head pose를 바꾸면 출력이 바뀌어야 한다 — 입력이 실제로 연결됐는지 확인."""
    common.layers.DTYPE = model.DTYPE = np.float32
    net = GazeNet(seed=0)
    image = RNG.standard_normal((16, *EYE_SHAPE)).astype(np.float32)
    head = RNG.standard_normal((16, 2)).astype(np.float32)

    a = net.predict(image, head)
    b = net.predict(image, head + 1.0)
    delta = np.abs(a - b).mean()
    print(f"\nhead pose를 1 rad 바꿨을 때 출력 변화: {delta:.4f}")
    assert delta > 1e-3, "head pose가 출력에 영향을 주지 않는다"


def test_can_overfit():
    """작은 묶음을 외울 수 있어야 한다. 학습 코드가 전혀 안 도는 경우를 잡는다."""
    from common.optimizer import Adam

    common.layers.DTYPE = model.DTYPE = np.float32
    net = GazeNet(conv_channels=(8, 16), hidden=32, joint=16, dropout=0.0, seed=0)
    opt = Adam(lr=3e-3)

    image = RNG.standard_normal((32, *EYE_SHAPE)).astype(np.float32)
    head = RNG.standard_normal((32, 2)).astype(np.float32)
    gaze = RNG.standard_normal((32, 2)).astype(np.float32)

    first = net.loss(image, head, gaze)
    for _ in range(150):
        opt.update(net.params, net.gradient(image, head, gaze))
    last = net.loss(image, head, gaze)
    print(f"32개 과적합: 손실 {first:.4f} → {last:.6f}")
    assert last < first * 0.01, "작은 묶음도 외우지 못한다"


def test_adversarial_gradient():
    """GRL이 붙은 줄기의 기울기는 손실의 참 기울기가 **아니다**. 그게 핵심이다.

        줄기로 가는 기울기 = ∂L_task/∂θ − scale · ∂L_adv/∂θ

    분류기는 사람을 맞히려 내려가고, 줄기는 그 반대로 밀린다. 그래서 전체 손실에
    대한 수치 미분과 대조하면 당연히 어긋난다. 위 식 그대로인지를 확인한다.
    """
    import common.layers
    import model as model_mod

    print("\n=== GRL 줄기 기울기 = ∂L_task − scale·∂L_adv 인지 ===")
    common.layers.DTYPE = model_mod.DTYPE = np.float64
    scale = 0.7

    net = GazeNet(conv_channels=(4, 6), hidden=12, joint=8, dropout=0.0,
                  n_persons=5, adv_scale=scale, seed=0)
    n = 6
    image = RNG.standard_normal((n, *EYE_SHAPE))
    head = RNG.standard_normal((n, 2))
    gaze = RNG.standard_normal((n, 2))
    person = RNG.integers(0, 5, n)

    analytic = net.gradient(image, head, gaze, person=person)
    scale_ref = max(np.abs(v).max() for v in analytic.values())

    def num_grad(param, picks, **loss_kwargs):
        flat = param.ravel()
        out = np.empty(len(picks))
        for j, i in enumerate(picks):
            orig = flat[i]
            flat[i] = orig + 1e-6
            f1 = net.loss(image, head, gaze, train_flg=True, **loss_kwargs)
            flat[i] = orig - 1e-6
            f2 = net.loss(image, head, gaze, train_flg=True, **loss_kwargs)
            flat[i] = orig
            out[j] = (f1 - f2) / 2e-6
        return out

    worst = 0.0
    for key, param in net.params.items():
        flat = param.ravel()
        picks = RNG.choice(flat.size, size=min(5, flat.size), replace=False)
        task = num_grad(param, picks)                      # person 없이 = L_task만
        total = num_grad(param, picks, person=person)      # L_task + L_adv
        adv = total - task

        trunk = not key.startswith("adv")
        expected = task - scale * adv if trunk else total
        got = analytic[key].ravel()[picks]
        err = grad_err(got, expected, scale_ref)
        worst = max(worst, err)
        print(f"  {key:<12} {'줄기' if trunk else '분류기'}  err {err:.2e}")

    # ∂L_adv를 두 수치 미분의 차(total − task)로 얻으므로 오차가 두 배로 쌓인다.
    # 직접 비교하는 test_gradient보다 기준을 한 자리 느슨하게 둔다.
    print(f"\n최악 {worst:.2e}  (기준 1e-4 — 수치 미분 두 개의 차라 오차가 겹친다)")
    assert worst < 1e-4, "GRL 기울기가 정의와 다르다"


def test_groupnorm_has_no_running_state():
    """GN은 배치에도 running stats에도 기대지 않는다 — LOPO에서 쓰는 이유다."""
    import common.layers
    import model as model_mod

    common.layers.DTYPE = model_mod.DTYPE = np.float32
    net = GazeNet(norm="group", dropout=0.0, seed=0)
    image = RNG.standard_normal((16, *EYE_SHAPE)).astype(np.float32)
    head = RNG.standard_normal((16, 2)).astype(np.float32)

    train_out = net.predict(image, head, train_flg=True)
    eval_out = net.predict(image, head, train_flg=False)
    same_mode = float(np.abs(train_out - eval_out).max())

    # 배치를 반으로 갈라도 각 샘플의 출력이 그대로여야 한다 (BN이라면 달라진다)
    half = np.concatenate([net.predict(image[:8], head[:8], train_flg=True),
                           net.predict(image[8:], head[8:], train_flg=True)])
    batch_free = float(np.abs(train_out - half).max())

    print(f"\nGroupNorm  학습/추론 차이 {same_mode:.2e}, 배치를 갈랐을 때 차이 {batch_free:.2e}")
    assert same_mode < 1e-5 and batch_free < 1e-4
    assert not any("running" in k for k in net.state_dict()), "GN에 running 상태가 있다"


def test_throughput(batch_size=128, steps=12):
    common.layers.DTYPE = model.DTYPE = np.float32
    net = GazeNet(seed=0)
    image = RNG.standard_normal((batch_size, *EYE_SHAPE)).astype(np.float32)
    head = RNG.standard_normal((batch_size, 2)).astype(np.float32)
    gaze = RNG.standard_normal((batch_size, 2)).astype(np.float32)

    net.gradient(image, head, gaze)  # 워밍업
    t = time.perf_counter()
    for _ in range(steps):
        net.gradient(image, head, gaze)
    per_step = (time.perf_counter() - t) / steps

    rate = batch_size / per_step
    print(f"\n학습 1스텝(batch {batch_size}): {per_step * 1000:.1f} ms "
          f"→ {rate:,.0f} samples/s")
    print(f"42,000개 1 epoch 추정: {42_000 / rate:.1f} s")


if __name__ == "__main__":
    test_gradient()
    test_adversarial_gradient()
    test_groupnorm_has_no_running_state()
    test_head_pose_is_used()
    test_can_overfit()
    test_throughput()
    print("\n전부 통과.")
