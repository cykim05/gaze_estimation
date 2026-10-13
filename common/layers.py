"""계층 — 교재 『밑바닥부터 시작하는 딥러닝』 common/layers.py.

모든 계층은 교재와 같은 규약을 지킨다.

    forward(x)   순전파. 역전파에 필요한 중간값을 self에 남긴다
    backward(dout) 입력에 대한 기울기를 돌려주고, 파라미터 기울기는 self.dW/self.db에 쌓는다

교재와 다른 점은 두 가지다.
  - dtype을 float32로 고정한다. BLAS sgemm이 dgemm의 2배라 numpy CNN에서는 이게 전부다.
  - BatchNormalization이 4D 입력을 채널 단위로 정규화한다(아래 docstring).
회귀 과제이므로 SoftmaxWithLoss 대신 MeanSquaredError를 쓴다.
"""

import numpy as np

from common.util import col2im, conv_output_size, im2col

DTYPE = np.float32


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class Affine:
    """완전연결. 교재와 같이 4D 입력을 받으면 알아서 펴므로 Flatten 계층이 따로 없다."""

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    @property
    def params(self):
        return {"W": self.W, "b": self.b}

    @property
    def grads(self):
        return {"W": self.dW, "b": self.db}

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx.reshape(*self.original_x_shape)


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    @property
    def params(self):
        return {"W": self.W, "b": self.b}

    @property
    def grads(self):
        return {"W": self.dW, "b": self.db}

    def forward(self, x):
        FN, _, FH, FW = self.W.shape
        N, _, H, W = x.shape
        out_h = conv_output_size(H, FH, self.stride, self.pad)
        out_w = conv_output_size(W, FW, self.stride, self.pad)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        self.x = x
        self.col = col
        self.col_W = col_W
        return out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout).transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)


class Pooling:
    """맥스 풀링. 역전파는 순전파에서 고른 위치로만 기울기를 보낸다."""

    def __init__(self, pool_h, pool_w, stride=None, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride if stride is not None else pool_h
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = conv_output_size(H, self.pool_h, self.stride, self.pad)
        out_w = conv_output_size(W, self.pool_w, self.stride, self.pad)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.x = x
        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        return out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1).ravel()
        pool_size = self.pool_h * self.pool_w

        dmax = np.zeros((dout.size, pool_size), dtype=dout.dtype)
        dmax[np.arange(dout.size), self.arg_max] = dout
        dcol = dmax.reshape(dout.size // self.x.shape[1], -1)
        return col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)


class BatchNormalization:
    """배치 정규화.

    교재는 4D 입력을 (N, C*H*W)로 펴서 픽셀 위치마다 따로 정규화한다. 여기서는
    conv 특징 맵을 채널 단위로 정규화한다(통계 C개). 표준 spatial BN이고,
    파라미터가 C*H*W개에서 C개로 줄어 사람 15명뿐인 이 데이터에서 과적합이 덜하다.
    2D 입력에서는 교재와 완전히 같다.
    """

    def __init__(self, size, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(size, dtype=DTYPE)
        self.beta = np.zeros(size, dtype=DTYPE)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(size, dtype=DTYPE)
        self.running_var = np.ones(size, dtype=DTYPE)
        self.dgamma = None
        self.dbeta = None
        self._shape = None
        self._xc = None
        self._xn = None
        self._std = None

    @property
    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    @property
    def grads(self):
        return {"gamma": self.dgamma, "beta": self.dbeta}

    @property
    def state(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    @staticmethod
    def _to2d(x):
        """(N, C, H, W) → (N*H*W, C). 2D는 그대로. 채널이 마지막 축으로 온다."""
        if x.ndim == 2:
            return x
        return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])

    @staticmethod
    def _from2d(out, shape):
        if len(shape) == 2:
            return out
        N, C, H, W = shape
        return out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    def forward(self, x, train_flg=True):
        self._shape = x.shape
        xf = self._to2d(x)

        if train_flg:
            mu = xf.mean(axis=0)
            xc = xf - mu
            var = (xc * xc).mean(axis=0)
            std = np.sqrt(var + self.eps)
            xn = xc / std

            self._xc, self._xn, self._std = xc, xn, std
            self.running_mean *= self.momentum
            self.running_mean += (1 - self.momentum) * mu
            self.running_var *= self.momentum
            self.running_var += (1 - self.momentum) * var
        else:
            xc = xf - self.running_mean
            xn = xc / np.sqrt(self.running_var + self.eps)

        out = self.gamma * xn + self.beta
        return self._from2d(out, self._shape)

    def backward(self, dout):
        doutf = self._to2d(dout)
        n = doutf.shape[0]

        self.dbeta = doutf.sum(axis=0)
        self.dgamma = (self._xn * doutf).sum(axis=0)

        dxn = self.gamma * doutf
        dxc = dxn / self._std
        dstd = -((dxn * self._xc) / (self._std * self._std)).sum(axis=0)
        dvar = 0.5 * dstd / self._std
        dxc += (2.0 / n) * self._xc * dvar
        dx = dxc - dxc.mean(axis=0)

        return self._from2d(dx, self._shape)


class Dropout:
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.mask = None
        self.rng = np.random.default_rng()

    def forward(self, x, train_flg=True):
        if not train_flg:
            return x
        self.mask = (self.rng.random(x.shape) > self.ratio).astype(x.dtype)
        return x * self.mask / (1.0 - self.ratio)

    def backward(self, dout):
        return dout * self.mask / (1.0 - self.ratio)


class MeanSquaredError:
    """회귀용 손실. 샘플당 0.5*||y-t||^2의 배치 평균. SoftmaxWithLoss 자리를 대신한다."""

    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y = y
        self.t = t
        diff = y - t
        return 0.5 * float(np.sum(diff * diff)) / y.shape[0]

    def backward(self, dout=1.0):
        return dout * (self.y - self.t) / self.y.shape[0]
