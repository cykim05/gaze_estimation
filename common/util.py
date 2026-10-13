"""im2col / col2im — 교재 『밑바닥부터 시작하는 딥러닝』 common/util.py.

합성곱을 행렬곱 한 번으로 바꾸는 변환이다. numpy만으로 45,000장을 돌리려면
계산을 전부 BLAS sgemm에 넘겨야 하므로 여기가 성능의 전부다.
교재 원본과 다른 점은 dtype을 입력 그대로 유지하는 것뿐이다(float32 유지 → 2배 빠름).
"""

import numpy as np


def conv_output_size(size: int, filter_size: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - filter_size) // stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """(N, C, H, W) → (N*out_h*out_w, C*filter_h*filter_w)."""
    N, C, H, W = input_data.shape
    out_h = conv_output_size(H, filter_h, stride, pad)
    out_w = conv_output_size(W, filter_w, stride, pad)

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.empty((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """im2col의 역변환. 겹치는 위치는 더한다(기울기 누적)."""
    N, C, H, W = input_shape
    out_h = conv_output_size(H, filter_h, stride, pad)
    out_w = conv_output_size(W, filter_w, stride, pad)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
