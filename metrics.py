"""평가 지표와 결과 표 — README의 정의를 그대로 옮긴 곳.

각도 오차는 (yaw, pitch)를 3D 단위 벡터로 되돌린 뒤 두 벡터의 사잇각을 도로 잰다.
yaw/pitch를 따로 재거나 2D 거리로 재면 다른 수가 나오므로 여기 한 곳만 쓴다.
"""

import numpy as np

from dataset import PERSONS
from preprocess.mpii_norm import gaze_to_3d


def angular_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """pred, gt: (N, 2) (yaw, pitch) 라디안 → (N,) 도."""
    a, b = gaze_to_3d(pred), gaze_to_3d(gt)
    cos = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def mean_angular_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(angular_error(pred, gt).mean())


def fold_table(name: str, errors: dict[str, float], persons=PERSONS) -> str:
    """README의 보고 형식(markdown 한 줄)으로 fold별 결과를 찍는다."""
    cells = [f"{errors[p]:.2f}" if p in errors else "" for p in persons]
    done = [errors[p] for p in persons if p in errors]
    mean = f"**{np.mean(done):.2f}**" if len(done) == len(persons) else f"{np.mean(done):.2f}*"
    return f"| {name} | " + " | ".join(cells) + f" | {mean} |"


def fold_table_header(persons=PERSONS) -> str:
    head = "| 방법 | " + " | ".join(persons) + " | 평균 |"
    rule = "|------|" + "-----|" * len(persons) + "------|"
    return f"{head}\n{rule}"


def summary(errors: dict[str, float]) -> str:
    values = np.array(list(errors.values()))
    return (f"mean {values.mean():.3f}°  std {values.std():.3f}°  "
            f"min {values.min():.2f}° ({min(errors, key=errors.get)})  "
            f"max {values.max():.2f}° ({max(errors, key=errors.get)})")
