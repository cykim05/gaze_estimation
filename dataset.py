"""데이터 적재 — reader.py 위에 올린 numpy 캐시 계층.

reader.py는 샘플마다 cv2.imread를 한다. leave-one-person-out은 같은 45,000장을
15 fold × epoch 수만큼 다시 읽게 되므로, 한 번만 디코딩해 배열로 굳혀 두고
그 뒤로는 메모리에서만 꺼낸다. uint8로 보관하면 이미지 전체가 93MB다.

    samples = load()                 # 15명 전부 (처음 한 번만 jpg를 읽는다)
    train = samples.exclude("p00")   # 14명 42,000개
    test = samples.only("p00")       #  1명  3,000개
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

import reader

PERSONS = tuple(f"p{k:02d}" for k in range(15))
DATA_ROOT = "data/EyeBased/MPIIGaze"
CACHE_DIR = "data/cache"

EYE_SHAPE = (1, 36, 60)


@dataclass(frozen=True)
class Samples:
    """사람 단위로 자를 수 있는 샘플 묶음. 모든 배열의 첫 축이 샘플이다."""

    image: np.ndarray   # (N, 1, 36, 60) uint8  — 0~255 grayscale
    head: np.ndarray    # (N, 2) float32 — head pose (yaw, pitch) rad
    gaze: np.ndarray    # (N, 2) float32 — 정답 gaze (yaw, pitch) rad
    person: np.ndarray  # (N,) uint8 — PERSONS 인덱스

    def __len__(self) -> int:
        return len(self.image)

    def take(self, index) -> "Samples":
        return replace(
            self,
            image=self.image[index],
            head=self.head[index],
            gaze=self.gaze[index],
            person=self.person[index],
        )

    def only(self, *persons: str) -> "Samples":
        return self.take(self._mask(persons))

    def exclude(self, *persons: str) -> "Samples":
        return self.take(~self._mask(persons))

    def _mask(self, persons) -> np.ndarray:
        wanted = [PERSONS.index(p) for p in persons]
        return np.isin(self.person, wanted)

    @property
    def person_names(self) -> list[str]:
        return [PERSONS[i] for i in np.unique(self.person)]

    @property
    def person_array(self) -> np.ndarray:
        """샘플마다 사람 이름. 다른 곳에서 만든 배열과 순서를 맞출 때 쓴다."""
        return np.asarray(PERSONS)[self.person]

    @staticmethod
    def concat(parts: list["Samples"]) -> "Samples":
        return Samples(
            image=np.concatenate([p.image for p in parts]),
            head=np.concatenate([p.head for p in parts]),
            gaze=np.concatenate([p.gaze for p in parts]),
            person=np.concatenate([p.person for p in parts]),
        )


def _read_person(person: str, root: str) -> Samples:
    """reader.loader로 한 사람분을 읽어 배열로 쌓는다."""
    label_path = os.path.join(root, "Label", f"{person}.label")
    dataset = reader.loader(label_path, os.path.join(root, "Image"))
    n = len(dataset)

    image = np.empty((n, *EYE_SHAPE), dtype=np.uint8)
    head = np.empty((n, 2), dtype=np.float32)
    gaze = np.empty((n, 2), dtype=np.float32)

    for i in range(n):
        info, label = dataset[i]
        # reader는 0~1 float으로 돌려준다. 캐시는 uint8로 되돌려 보관한다.
        image[i] = np.rint(info["eye"] * 255.0).astype(np.uint8)
        head[i] = info["head_pose"]
        gaze[i] = label

    return Samples(
        image=image,
        head=head,
        gaze=gaze,
        person=np.full(n, PERSONS.index(person), dtype=np.uint8),
    )


def load(persons=PERSONS, root=DATA_ROOT, cache_dir=CACHE_DIR, verbose=True) -> Samples:
    """사람별 캐시(.npz)를 읽고, 없으면 jpg에서 만들어 저장한다."""
    os.makedirs(cache_dir, exist_ok=True)
    parts = []

    for person in persons:
        cache = os.path.join(cache_dir, f"{person}.npz")
        if os.path.exists(cache):
            with np.load(cache) as z:
                parts.append(Samples(z["image"], z["head"], z["gaze"], z["person"]))
        else:
            if verbose:
                print(f"[cache] building {cache}")
            part = _read_person(person, root)
            np.savez(cache, image=part.image, head=part.head, gaze=part.gaze, person=part.person)
            parts.append(part)

    out = Samples.concat(parts)
    if verbose:
        print(f"[data] {len(out)} samples, {len(out.person_names)} persons, "
              f"image {out.image.shape[1:]} {out.image.dtype}")
    return out


class Standardizer:
    """평균 0, 표준편차 1로 맞춘다. 반드시 학습 fold로만 fit한다(테스트 사람 누설 금지)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x: np.ndarray, axis=0, eps=1e-8) -> "Standardizer":
        mean = x.mean(axis=axis, keepdims=True).astype(np.float32)
        std = x.std(axis=axis, keepdims=True).astype(np.float32) + np.float32(eps)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


class Augment:
    """학습 배치에만 거는 증강. 추론·평가 경로에는 걸지 않는다.

    **좌우 반전은 쓸 수 없다.** 오른눈은 전처리에서 이미 왼눈처럼 뒤집혀 저장돼
    있다. 왼눈 이미지를 다시 뒤집으면 데이터에 없는 영상이 된다.

    쓸 수 있는 것은 셋이다.
    - shift: 눈 중심 검출이 몇 픽셀 틀렸을 때를 흉내 낸다. 정규화가 눈을 가운데
      맞춰 놓은 탓에 모델이 '가운데가 동공'이라는 가정에 기대기 쉽다(EDA 5절).
    - brightness/contrast: histogram equalization이 지우고 남은 조명 차이를 넓힌다.
    - cutout: 눈의 일부를 가려 한 군데(예: 속눈썹 모양)에만 의존하지 못하게 한다.
      사람이 바뀌면 그 한 군데가 제일 먼저 달라진다.
    """

    def __init__(self, shift=2, brightness=0.12, contrast=0.12, cutout=8,
                 cutout_prob=0.5, seed=None):
        self.shift = shift
        self.brightness = brightness
        self.contrast = contrast
        self.cutout = cutout
        self.cutout_prob = cutout_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """(N, 1, 36, 60) uint8 → 같은 모양 float32 (0~255 눈금 유지)."""
        n, _, h, w = image.shape
        rng = self.rng

        if self.shift:
            s = self.shift
            padded = np.pad(image, ((0, 0), (0, 0), (s, s), (s, s)), mode="edge")
            dy = rng.integers(0, 2 * s + 1, n)
            dx = rng.integers(0, 2 * s + 1, n)
            shifted = np.empty_like(image)
            for i in range(n):
                shifted[i] = padded[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w]
            image = shifted

        out = image.astype(np.float32)

        if self.brightness or self.contrast:
            gain = 1.0 + rng.uniform(-self.contrast, self.contrast, (n, 1, 1, 1))
            bias = rng.uniform(-self.brightness, self.brightness, (n, 1, 1, 1)) * 255.0
            out = (out - 127.5) * gain.astype(np.float32) + 127.5 + bias.astype(np.float32)
            np.clip(out, 0.0, 255.0, out=out)

        if self.cutout:
            hit = rng.random(n) < self.cutout_prob
            cy = rng.integers(0, h, n)
            cx = rng.integers(0, w, n)
            half = self.cutout // 2
            for i in np.flatnonzero(hit):
                y0, y1 = max(0, cy[i] - half), min(h, cy[i] + half)
                x0, x1 = max(0, cx[i] - half), min(w, cx[i] + half)
                out[i, :, y0:y1, x0:x1] = 127.5

        return out


class Batches:
    """미니배치 반복자. uint8 이미지를 배치 단위로만 float32로 바꾼다.

    45,000장을 float32로 한꺼번에 들고 있으면 389MB지만, 배치마다 바꾸면
    캐시에 들어가는 크기로 유지된다.
    """

    def __init__(self, samples: Samples, batch_size: int, image_norm, head_std: Standardizer,
                 gaze_std: Standardizer | None = None, shuffle=True, drop_last=False,
                 augment: "Augment | None" = None, seed=None):
        self.samples = samples
        self.batch_size = batch_size
        self.image_norm = image_norm
        self.head_std = head_std
        self.gaze_std = gaze_std
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.samples)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.samples)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)

        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            raw = self.samples.image[idx]
            image = self.image_norm(self.augment(raw) if self.augment else raw)
            head = self.head_std.transform(self.samples.head[idx])
            gaze = self.samples.gaze[idx]
            yield image, head, (self.gaze_std.transform(gaze) if self.gaze_std else gaze)


class ImageNorm:
    """0~255 uint8 → 평균 0, 표준편차 1 float32. 통계는 스칼라 하나씩이다.

    픽셀 위치별 통계를 쓰지 않는 이유는 눈 이미지가 이미 perspective normalization과
    histogram equalization을 거쳐 위치가 정렬돼 있어서, 위치별 평균을 빼면
    학습 fold에 있던 사람들의 눈 모양을 그대로 빼는 셈이 되기 때문이다.
    """

    def __init__(self, mean: float, std: float):
        self.mean = np.float32(mean)
        self.std = np.float32(std)

    @classmethod
    def fit(cls, image: np.ndarray, sample=100_000, seed=0) -> "ImageNorm":
        flat = image.reshape(-1)
        if flat.size > sample:  # 통계는 일부 픽셀만 봐도 충분하다
            flat = flat[np.random.default_rng(seed).integers(0, flat.size, sample)]
        flat = flat.astype(np.float32) / 255.0
        return cls(float(flat.mean()), float(flat.std()) + 1e-8)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """uint8도 Augment가 내놓은 float32(0~255 눈금)도 그대로 받는다."""
        return (image.astype(np.float32) / np.float32(255.0) - self.mean) / self.std


if __name__ == "__main__":
    samples = load()
    print("head  mean/std:", samples.head.mean(0).round(4), samples.head.std(0).round(4))
    print("gaze  mean/std:", samples.gaze.mean(0).round(4), samples.gaze.std(0).round(4))
    print("p00만:", len(samples.only("p00")), " p00 제외:", len(samples.exclude("p00")))


def label_frame(persons=PERSONS, root=DATA_ROOT):
    """라벨 파일의 모든 칸을 pandas DataFrame으로 — EDA 전용.

    학습에는 Image/2DGaze/2DHead만 쓰지만(README), 어떤 데이터인지 보려면
    WhichEye·Origin(day)·GazeOrigin(눈~카메라 거리) 같은 칸이 필요하다.
    """
    import pandas as pd

    vector_cols = {"3DHead": 3, "2DGaze": 2, "2DHead": 2, "Rmat": 3, "Smat": 3, "GazeOrigin": 3}
    axes = {2: ("yaw", "pitch"), 3: ("x", "y", "z")}
    rows = []

    for person in persons:
        path = os.path.join(root, "Label", f"{person}.label")
        with open(path) as f:
            header = f.readline().split()
            for line in f:
                if not line.strip():
                    continue
                cells = dict(zip(header, line.split()))
                day, frame = cells["Origin"].split("/")
                row = {
                    "person": person,
                    "image": cells["Image"],
                    "day": day,
                    "frame": int(os.path.splitext(frame)[0]),
                    "which_eye": cells["WhichEye"],
                }
                for col, dim in vector_cols.items():
                    values = np.asarray(cells[col].split(","), dtype=np.float64)
                    for axis, value in zip(axes[dim], values):
                        row[f"{col}_{axis}"] = value
                rows.append(row)

    return pd.DataFrame(rows)
