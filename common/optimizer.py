"""갱신 방법 — 교재 『밑바닥부터 시작하는 딥러닝』 common/optimizer.py.

교재와 같이 update(params, grads)가 dict를 받아 params를 제자리에서 갱신한다.
키는 모델이 정하고, 옵티마이저는 키를 그대로 따라간다.
"""

import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        for key in params:
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {k: np.zeros_like(v) for k, v in params.items()}
        for key in params:
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """Kingma & Ba 2015. 교재 구현과 같고, lr을 밖에서 바꿀 수 있게 열어 두었다."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.iter += 1
        bias1 = 1.0 - self.beta1 ** self.iter
        bias2 = 1.0 - self.beta2 ** self.iter

        for key in params:
            g = grads[key]
            self.m[key] += (1 - self.beta1) * (g - self.m[key])
            self.v[key] += (1 - self.beta2) * (g * g - self.v[key])
            m_hat = self.m[key] / bias1
            v_hat = self.v[key] / bias2
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
