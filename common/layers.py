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
    """교재는 불 마스크로 fancy indexing을 하지만, 큰 특징 맵에서는 그 쓰기가 비싸다.
    np.maximum 한 번으로 같은 값을 얻고 역전파는 출력의 부호만 본다."""

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.maximum(x, 0)
        return self.out

    def backward(self, dout):
        return dout * (self.out > 0)


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
        self.x_shape = None
        self.tie_mask = None
        self.tie_count = None
        # 창이 서로 겹치지 않고 패딩도 없으면 im2col 없이 reshape만으로 끝난다
        self._tiled = self.stride == pool_h == pool_w and pad == 0

    def forward(self, x):
        if self._tiled:
            return self._forward_tiled(x)

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
        if self._tiled:
            return self._backward_tiled(dout)

        dout = dout.transpose(0, 2, 3, 1).ravel()
        pool_size = self.pool_h * self.pool_w

        dmax = np.zeros((dout.size, pool_size), dtype=dout.dtype)
        dmax[np.arange(dout.size), self.arg_max] = dout
        dcol = dmax.reshape(dout.size // self.x.shape[1], -1)
        return col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

    # ------------------------------------------------- 겹치지 않는 풀링 전용 빠른 길

    def _forward_tiled(self, x):
        """창이 겹치지 않으면 im2col이 필요 없다. reshape는 뷰라서 복사가 0번이다.

        (N, C, H, W) → (N, C, H/ph, ph, W/pw, pw) 로 보고 ph·pw 축만 max한다.
        im2col 경로는 같은 일에 큰 배열을 네 번 복사한다.
        """
        N, C, H, W = x.shape
        ph, pw = self.pool_h, self.pool_w
        out_h, out_w = H // ph, W // pw

        view = x[:, :, :out_h * ph, :out_w * pw].reshape(N, C, out_h, ph, out_w, pw)
        out = view.max(axis=(3, 5))

        self.x_shape = x.shape
        # 최댓값이 난 자리. 동점이면 기울기를 n등분한다(ReLU 뒤의 0 동점이 흔하다).
        self.tie_mask = view == out[:, :, :, None, :, None]
        self.tie_count = self.tie_mask.sum(axis=(3, 5), keepdims=True)
        return out

    def _backward_tiled(self, dout):
        N, C, H, W = self.x_shape
        ph, pw = self.pool_h, self.pool_w
        out_h, out_w = H // ph, W // pw

        share = (dout / self.tie_count[:, :, :, 0, :, 0])[:, :, :, None, :, None]
        dview = self.tie_mask * share

        if (out_h * ph, out_w * pw) == (H, W):
            return dview.reshape(N, C, H, W)
        dx = np.zeros(self.x_shape, dtype=dout.dtype)  # 가장자리를 잘라 냈던 경우
        dx[:, :, :out_h * ph, :out_w * pw] = dview.reshape(N, C, out_h * ph, out_w * pw)
        return dx


class BatchNormalization:
    """배치 정규화.

    교재는 4D 입력을 (N, C*H*W)로 펴서 픽셀 위치마다 따로 정규화한다. 여기서는
    conv 특징 맵을 채널 단위로 정규화한다(통계 C개). 표준 spatial BN이고,
    파라미터가 C*H*W개에서 C개로 줄어 사람 15명뿐인 이 데이터에서 과적합이 덜하다.
    2D 입력에서는 교재와 완전히 같다.

    4D를 2D로 폈다 되돌리면 큰 배열을 네 번 복사하게 되므로, 축을 그대로 두고
    (0, 2, 3)축으로만 평균을 낸다. 복사가 없다.
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
        self._axes_cache = None
        self._n = None
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
    def _axes(ndim):
        """정규화할 축과 gamma/beta를 끼워 넣을 모양. 통계는 언제나 채널(축 1)마다 하나다."""
        if ndim == 2:
            return (0,), (1, -1)
        return (0, 2, 3), (1, -1, 1, 1)

    def forward(self, x, train_flg=True):
        axes, bshape = self._axes(x.ndim)
        self._axes_cache = axes

        if train_flg:
            mu = x.mean(axis=axes, keepdims=True)
            xc = x - mu
            var = (xc * xc).mean(axis=axes, keepdims=True)
            std = np.sqrt(var + self.eps)
            xn = xc / std

            self._xc, self._xn, self._std = xc, xn, std
            self._n = x.size // x.shape[1] if x.ndim == 4 else x.shape[0]

            self.running_mean *= self.momentum
            self.running_mean += (1 - self.momentum) * mu.reshape(-1)
            self.running_var *= self.momentum
            self.running_var += (1 - self.momentum) * var.reshape(-1)
        else:
            mean = self.running_mean.reshape(bshape)
            std = np.sqrt(self.running_var.reshape(bshape) + self.eps)
            xn = (x - mean) / std

        return self.gamma.reshape(bshape) * xn + self.beta.reshape(bshape)

    def backward(self, dout):
        """계산 그래프를 한 단계씩 되짚는 대신 정리된 꼴을 쓴다.

            dx = (gamma / std) * (dout - dbeta/n - xn * dgamma/n)

        교재의 단계별 역전파와 수학적으로 같지만, 특징 맵만 한 임시 배열을
        열 번 만드는 대신 네 번만 만든다. 큰 특징 맵에서는 이게 그대로 시간이다.
        """
        axes = self._axes_cache
        bshape = self._axes(dout.ndim)[1]

        self.dbeta = dout.sum(axis=axes)
        self.dgamma = (self._xn * dout).sum(axis=axes)

        dx = dout - (self.dbeta / self._n).reshape(bshape)
        dx -= self._xn * (self.dgamma / self._n).reshape(bshape)
        dx *= self.gamma.reshape(bshape) / self._std
        return dx


class Dropout:
    """seed를 받는다. 받지 않으면 실행마다 결과가 달라져 재현이 안 된다."""

    def __init__(self, ratio=0.5, seed=None):
        self.ratio = ratio
        self.mask = None
        self.rng = np.random.default_rng(seed)

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
