"""배열 모듈 고르기 — numpy(CPU) 또는 cupy(GPU).

교재 코드는 전부 배열 연산이라, 배열 모듈만 바꾸면 같은 계층 코드가 GPU에서 돈다.
딥러닝 프레임워크를 들이는 것이 아니라 numpy 자리에 cupy를 끼우는 것뿐이다.
`common/layers.py`는 `from common.backend import xp as np`로 받아 코드가 한 줄도
달라지지 않는다.

장치는 import 시점에 `GAZE_DEVICE` 환경변수로 정한다. 학습은 어차피 프로세스를
새로 띄우므로 실행 중에 바꿀 일이 없고, 전역 상태를 쓰지 않는 쪽이 안전하다.

    GAZE_DEVICE=gpu python train.py --device gpu
"""

from __future__ import annotations

import os

import numpy

DEVICE = os.environ.get("GAZE_DEVICE", "cpu").lower()

if DEVICE == "gpu":
    try:
        import cupy
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "GAZE_DEVICE=gpu인데 cupy가 없다.\n"
            "  pip install cupy-cuda12x   (CUDA 12.x 기준)"
        ) from exc
    xp = cupy
else:
    cupy = None
    xp = numpy


def on_gpu() -> bool:
    return xp is not numpy


def to_device(array):
    """numpy 배열을 현재 장치로 올린다. 이미 맞으면 그대로 둔다."""
    if not on_gpu():
        return numpy.asarray(array)
    return xp.asarray(array)


def to_cpu(array) -> numpy.ndarray:
    """어떤 장치의 배열이든 numpy로 내린다. 평가·저장·그림은 전부 CPU에서 한다."""
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return numpy.asarray(array)


def synchronize() -> None:
    """GPU는 커널이 비동기로 돌아 시간을 재기 전에 기다려야 한다."""
    if on_gpu():
        xp.cuda.runtime.deviceSynchronize()


def describe() -> str:
    if not on_gpu():
        return f"CPU (numpy {numpy.__version__})"
    dev = xp.cuda.Device()
    free, total = xp.cuda.runtime.memGetInfo()
    name = xp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
    return (f"GPU {dev.id} {name} (cupy {xp.__version__}, "
            f"여유 {free / 2**30:.1f}/{total / 2**30:.1f} GiB)")
