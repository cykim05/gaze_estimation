"""GazeNet — 눈 이미지 CNN에 head pose를 붙여 gaze를 회귀하는 모델.

교재의 SimpleConvNet과 같은 뼈대다. 다른 점은 입력이 둘이라는 것뿐이다.
이미지는 conv 블록을 지나 특징 벡터가 되고, 거기에 head pose 2개를 이어 붙인 뒤
완전연결 두 층을 지나 (yaw, pitch)를 낸다. head pose를 처음부터 섞지 않는 이유는
2차원 숫자 두 개가 2,160픽셀과 같은 자리에서 경쟁해 봐야 묻히기 때문이다.

    conv 3×3 ×c1 → BN → ReLU → pool 2×2       (36×60 → 18×30)
    conv 3×3 ×c2 → BN → ReLU → pool 2×2       (18×30 →  9×15)
    conv 3×3 ×c3 → BN → ReLU → pool 2×2       ( 9×15 →  4× 7)
    flatten → Affine(hidden) → BN → ReLU → Dropout
    ⊕ head pose(2) → Affine(joint) → ReLU → Affine(2)

구조의 근거는 EDA다(eda.py 5절). 정규화로 눈 중심이 맞아 있어 변화는 동공 둘레의
국소 경계에 몰려 있다. 그래서 작은 3×3 커널과 세 번의 풀링으로 충분하다.
"""

from __future__ import annotations

import numpy

from common.backend import to_cpu, to_device
from common.backend import xp as np
from common.layers import (
    Affine,
    BatchNormalization,
    Convolution,
    Dropout,
    GradientReversal,
    GroupNormalization,
    Pooling,
    Relu,
    SoftmaxWithLoss,
    WeightedMeanSquaredError,
)

NORMS = {"batch": BatchNormalization, "group": GroupNormalization}

DTYPE = np.float32
EYE_SHAPE = (1, 36, 60)
HEAD_DIM = 2
OUT_DIM = 2


def he_normal(shape, fan_in, rng):
    """ReLU 앞단의 표준 초깃값. 교재의 '권장 가중치 초깃값'과 같다.

    난수는 반드시 CPU(numpy)에서 뽑아 장치로 올린다. cupy의 생성기는 같은 seed로도
    numpy와 다른 수열을 내므로, 장치에서 바로 뽑으면 CPU와 GPU가 다른 모델이 된다.
    """
    return to_device((rng.standard_normal(shape) * numpy.sqrt(2.0 / fan_in)).astype(DTYPE))


class GazeNet:
    def __init__(self, conv_channels=(24, 48, 96), hidden=256, joint=96,
                 dropout=0.2, input_shape=EYE_SHAPE, norm="batch",
                 n_persons=0, adv_scale=1.0, adv_hidden=64, seed=None):
        rng = numpy.random.default_rng(seed)   # 초깃값은 장치와 무관해야 한다
        c_in, h, w = input_shape
        Norm = NORMS[norm]
        self.norm_name = norm

        self.image_layers: list = []
        self._named: list[tuple[str, object]] = []

        for i, c_out in enumerate(conv_channels, start=1):
            # 뒤에 BN이 오므로 편향을 두지 않는다. BN의 beta가 같은 일을 한다.
            W = he_normal((c_out, c_in, 3, 3), c_in * 9, rng)
            conv = Convolution(W, stride=1, pad=1)
            bn = Norm(c_out)
            self._add(f"conv{i}", conv)
            self._add(f"bn{i}", bn)
            self.image_layers += [conv, bn, Relu(), Pooling(2, 2)]
            c_in, h, w = c_out, h // 2, w // 2

        flat = c_in * h * w
        fc = Affine(he_normal((flat, hidden), flat, rng))   # 뒤가 정규화라 편향 없음
        bn_fc = Norm(hidden)
        self._add("fc", fc)
        self._add("bn_fc", bn_fc)
        # 가중치 초깃값과 다른 흐름을 쓰도록 seed를 갈라 준다
        self.image_layers += [fc, bn_fc, Relu(),
                              Dropout(dropout, seed=None if seed is None else seed + 1)]

        # head pose를 이어 붙인 뒤의 두 층. 마지막은 활성화 없이 그대로 (yaw, pitch).
        fuse = Affine(he_normal((hidden + HEAD_DIM, joint), hidden + HEAD_DIM, rng),
                      np.zeros(joint, dtype=DTYPE))
        out = Affine(he_normal((joint, OUT_DIM), joint, rng) * DTYPE(0.1),
                     np.zeros(OUT_DIM, dtype=DTYPE))
        self._add("fuse", fuse)
        self._add("out", out)
        self.joint_layers = [fuse, Relu(), out]

        # 사람 분류 head (domain-adversarial). GradientReversal이 맨 앞에 있어
        # 특징 추출기로 가는 기울기의 부호가 뒤집힌다. 분류기는 사람을 맞히려 하고
        # 특징 추출기는 못 맞히게 밀려, 특징에서 사람 정체성이 빠진다.
        self.person_layers: list = []
        self.person_loss = None
        if n_persons:
            pw1 = Affine(he_normal((hidden, adv_hidden), hidden, rng),
                         np.zeros(adv_hidden, dtype=DTYPE))
            pw2 = Affine(he_normal((adv_hidden, n_persons), adv_hidden, rng),
                         np.zeros(n_persons, dtype=DTYPE))
            self._add("adv1", pw1)
            self._add("adv2", pw2)
            self.person_layers = [GradientReversal(adv_scale), pw1, Relu(), pw2]
            self.person_loss = SoftmaxWithLoss()

        self.last_layer = WeightedMeanSquaredError()
        self.feature_dim = hidden
        self.params = {f"{name}.{k}": v
                       for name, layer in self._named for k, v in layer.params.items()}

    def _add(self, name: str, layer) -> None:
        self._named.append((name, layer))

    @staticmethod
    def _forward(layers, x, train_flg):
        for layer in layers:
            if isinstance(layer, (BatchNormalization, GroupNormalization, Dropout)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    @staticmethod
    def _backward(layers, dout):
        for layer in reversed(layers):
            dout = layer.backward(dout)
        return dout

    def predict(self, image, head, train_flg=False):
        self._feature = self._forward(self.image_layers, image, train_flg)
        fused = np.concatenate([self._feature, head], axis=1)
        return self._forward(self.joint_layers, fused, train_flg)

    def loss(self, image, head, gaze, train_flg=True, weights=None, person=None) -> float:
        total = self.last_layer.forward(self.predict(image, head, train_flg), gaze, weights)
        if person is not None and self.person_loss is not None:
            logits = self._forward(self.person_layers, self._feature, train_flg)
            self._adv = self.person_loss.forward(logits, person)
            total += self._adv
        return total

    def per_sample_loss(self):
        """GroupDRO가 사람별 손실을 모을 때 쓴다. loss() 직후에만 유효하다."""
        return self.last_layer.per_sample()

    def gradient(self, image, head, gaze, weights=None, person=None):
        """순전파 → 역전파 한 번. 파라미터 기울기를 params와 같은 키로 돌려준다."""
        self.loss(image, head, gaze, train_flg=True, weights=weights, person=person)

        dout = self._backward(self.joint_layers, self.last_layer.backward())
        # 이어 붙인 자리를 되돌린다. head pose는 입력이라 기울기를 버린다.
        dfeature = dout[:, :self.feature_dim]
        if person is not None and self.person_loss is not None:
            dfeature = dfeature + self._backward(self.person_layers,
                                                 self.person_loss.backward())
        self._backward(self.image_layers, dfeature)

        return {f"{name}.{k}": v
                for name, layer in self._named for k, v in layer.grads.items()}

    # ------------------------------------------------------------------ 저장/복원

    def state_dict(self) -> dict:
        """항상 CPU(numpy) 배열로 돌려준다. 저장과 비교는 장치와 무관해야 한다."""
        state = {k: to_cpu(v) for k, v in self.params.items()}
        for name, layer in self._named:
            if isinstance(layer, BatchNormalization):   # GN은 들고 다닐 상태가 없다
                state.update({f"{name}.{k}": to_cpu(v) for k, v in layer.state.items()})
        return state

    def load_state_dict(self, state: dict) -> None:
        for name, layer in self._named:
            for key, value in layer.params.items():
                value[...] = to_device(state[f"{name}.{key}"])
            if isinstance(layer, BatchNormalization):
                for key, value in layer.state.items():
                    value[...] = to_device(state[f"{name}.{key}"])

    def n_params(self) -> int:
        return sum(v.size for v in self.params.values())

    def describe(self) -> str:
        lines = [f"GazeNet  파라미터 {self.n_params():,}개"]
        for name, layer in self._named:
            shapes = " ".join(f"{k}{tuple(v.shape)}" for k, v in layer.params.items())
            lines.append(f"  {name:<8} {type(layer).__name__:<20} {shapes}")
        return "\n".join(lines)


if __name__ == "__main__":
    net = GazeNet(seed=0)
    print(net.describe())

    rng = np.random.default_rng(0)
    image = rng.standard_normal((8, *EYE_SHAPE)).astype(DTYPE)
    head = rng.standard_normal((8, HEAD_DIM)).astype(DTYPE)
    gaze = rng.standard_normal((8, OUT_DIM)).astype(DTYPE)

    print(f"\n출력 shape: {net.predict(image, head).shape}")
    print(f"초기 손실 : {net.loss(image, head, gaze):.4f}")
    grads = net.gradient(image, head, gaze)
    print(f"기울기 키 {len(grads)}개, 전부 params와 shape 일치: "
          f"{all(grads[k].shape == v.shape for k, v in net.params.items())}")
