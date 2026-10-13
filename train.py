"""leave-one-person-out 학습·평가 파이프라인. `python train.py`

fold k는 pk 3,000개를 테스트로, 나머지 14명을 학습으로 쓴다. 14명 중 2명은
안쪽 검증용으로 떼어 내 epoch를 고르는 데만 쓴다. 테스트 사람의 데이터는
학습·정규화 통계·epoch 선택 어디에도 들어가지 않는다.

    python train.py                       # 15 fold 전부, 병렬
    python train.py --folds p00 --jobs 1  # 한 fold만
    python train.py --epochs 20 --lr 2e-3

구조(GazeNet)와 옵티마이저 설정은 EDA 결론과 통상적인 기본값으로 미리 고정했다.
fold마다 데이터를 보고 고르는 것은 '몇 epoch에서 멈출까' 하나뿐이고, 그것도
그 fold의 학습용 14명 안에서만 정한다.
"""

from __future__ import annotations

import os
import sys

# BLAS 스레드 수는 numpy를 들이기 전에 정해야 한다. 아래 import보다 위에 있어야 하는 이유다.
# conv는 BLAS로 가지만 BN·ReLU·풀링 같은 원소별 연산은 단일 스레드라, 스레드를 늘리는 것보다
# fold를 프로세스로 나눠 동시에 돌리는 쪽이 훨씬 빠르다(측정: 15 worker에서 9.7배).
# spawn으로 뜨는 자식 프로세스도 이 파일을 다시 import하므로 같은 값이 걸린다.
# 텍스트 파일은 전부 encoding="utf-8"을 명시한다. 로케일 기본값에 기대지 않는다.
# SLURM에서 돌린 탐색 작업이 이것 때문에 죽었다. 한 프로세스 안에서 첫 후보는 한글
# 로그를 정상적으로 썼는데 두 번째 후보부터 'ascii' codec 오류가 났다. 즉 프로세스가
# 도는 중에 open()의 기본 인코딩이 바뀌었다. CUDA 쪽 네이티브 라이브러리가 로케일을
# 건드리는 것으로 의심되지만 확인하지는 못했다. 원인이 무엇이든 명시하면 사라진다.

# --device gpu는 여기서 환경변수를 세워야 한다. common.backend가 import 시점에 읽는다.
if "--device" in sys.argv:
    os.environ["GAZE_DEVICE"] = sys.argv[sys.argv.index("--device") + 1]

THREADS_PER_FOLD = os.environ.get("GAZE_THREADS", "2")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = THREADS_PER_FOLD

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Config:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 2e-3
    lr_final_ratio: float = 0.02   # 코사인 감쇠의 끝 지점
    dropout: float = 0.2
    weight_decay: float = 0.0      # 가중치 행렬에만. bias·BN 파라미터는 건드리지 않는다
    augment: bool = False
    conv_channels: tuple = (24, 48, 96)
    hidden: int = 256
    joint: int = 96
    n_val_persons: int = 2
    seed: int = 0


# weight decay를 걸지 않는 파라미터. bias와 BN의 gamma·beta를 0으로 당기면
# 정규화가 아니라 표현력만 깎인다.
NO_DECAY_SUFFIX = (".b", ".gamma", ".beta")


def apply_weight_decay(params: dict, lr: float, weight_decay: float) -> None:
    """decoupled weight decay. Adam의 적응 보폭과 섞이지 않게 갱신과 따로 건다."""
    if weight_decay <= 0:
        return
    factor = 1.0 - lr * weight_decay
    for key, value in params.items():
        if not key.endswith(NO_DECAY_SUFFIX):
            value *= factor


def split_persons(test: str, n_val: int, persons) -> tuple[list[str], list[str]]:
    """테스트 사람을 뺀 14명을 학습 12 + 검증 2로 가른다.

    검증 대상은 테스트 사람 바로 뒤의 사람들로 정한다. fold마다 다른 사람이
    검증을 맡아 특정인에 맞춰지는 것을 피하고, 규칙이 결정적이라 재현된다.
    """
    rest = [p for p in persons if p != test]
    start = persons.index(test)
    val = [persons[(start + 1 + i) % len(persons)] for i in range(n_val)]
    return [p for p in rest if p not in val], val


def cosine_lr(step: int, total: int, lr: float, final_ratio: float) -> float:
    return lr * (final_ratio + (1 - final_ratio) * 0.5 * (1 + np.cos(np.pi * step / total)))


def predict_all(net, samples, norms, batch_size=512) -> np.ndarray:
    """추론 전용 경로. 입력은 장치 위에서, 결과는 CPU numpy 라디안으로 낸다."""
    from common.backend import to_cpu

    image_norm, head_std, gaze_std = norms
    out = np.empty((len(samples), 2), dtype=np.float32)
    for start in range(0, len(samples), batch_size):
        sl = slice(start, start + batch_size)
        out[sl] = to_cpu(net.predict(image_norm(samples.image[sl]),
                                     head_std.transform(samples.head[sl]), train_flg=False))
    return gaze_std.inverse(out)


def train_fold(cfg: Config, test_person: str, log_dir: str | None = None,
               fold_dir: str | None = None, track_test: bool = True,
               verbose: bool = True) -> dict:
    """fold 하나를 처음부터 학습하고 테스트 각도 오차를 돌려준다.

    병렬로 돌리면 stdout이 뒤섞이므로, log_dir이 있으면 fold마다 제 파일에 쓴다.
    """
    import dataset
    from common.backend import describe, synchronize
    from common.optimizer import Adam
    from dataset import Augment, Batches, ImageNorm, PERSONS, Standardizer
    from metrics import mean_angular_error
    from model import GazeNet

    t0 = time.perf_counter()
    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = open(os.path.join(log_dir, f"{test_person}.log"), "w",
                        encoding="utf-8", buffering=1)

    def say(line: str) -> None:
        if log_file:
            log_file.write(line + "\n")
        if verbose:
            print(line, flush=True)

    all_samples = dataset.load(verbose=False)
    train_persons, val_persons = split_persons(test_person, cfg.n_val_persons, list(PERSONS))

    train_cpu = all_samples.only(*train_persons)
    val_cpu = all_samples.only(*val_persons)
    test_cpu = all_samples.only(test_person)

    # 정규화 통계는 학습 12명에서만 뽑는다. 검증·테스트 사람은 쳐다보지 않는다.
    norms = (ImageNorm.fit(train_cpu.image),
             Standardizer.fit(train_cpu.head),
             Standardizer.fit(train_cpu.gaze))
    image_norm, head_std, gaze_std = norms

    # 45,000장이 uint8로 93MB라 통째로 장치에 올려 둔다. 배치마다 옮기지 않는다.
    train, val, test = train_cpu.to_device(), val_cpu.to_device(), test_cpu.to_device()
    say(f"[{test_person}] {describe()}  학습 {len(train)} / 검증 {len(val)} / 테스트 {len(test)}")

    net = GazeNet(conv_channels=cfg.conv_channels, hidden=cfg.hidden,
                  joint=cfg.joint, dropout=cfg.dropout, seed=cfg.seed)
    opt = Adam(lr=cfg.lr)
    augment = Augment(seed=cfg.seed + 2) if cfg.augment else None
    batches = Batches(train, cfg.batch_size, image_norm, head_std, gaze_std,
                      shuffle=True, drop_last=True, augment=augment, seed=cfg.seed)

    total_steps = cfg.epochs * len(batches)
    history, best = [], {"val": np.inf, "epoch": -1, "state": None}
    step = 0

    for epoch in range(1, cfg.epochs + 1):
        running, n_batch = 0.0, 0
        for image, head, gaze in batches:
            opt.lr = cosine_lr(step, total_steps, cfg.lr, cfg.lr_final_ratio)
            running += net.loss(image, head, gaze, train_flg=True)
            opt.update(net.params, net.gradient(image, head, gaze))
            apply_weight_decay(net.params, opt.lr, cfg.weight_decay)
            step += 1
            n_batch += 1

        synchronize()
        val_err = mean_angular_error(predict_all(net, val, norms), val_cpu.gaze)
        # track_test=False면 테스트를 아예 계산하지 않는다. 설정을 고르는 동안
        # 테스트 숫자가 눈에 띄는 일 자체를 막는다.
        test_err = mean_angular_error(predict_all(net, test, norms), test_cpu.gaze) \
            if track_test else float("nan")
        history.append({"epoch": epoch, "loss": running / n_batch,
                        "val": val_err, "test": test_err})

        if val_err < best["val"]:
            best = {"val": val_err, "epoch": epoch, "state": net.state_dict()}
            best["state"] = {k: v.copy() for k, v in best["state"].items()}

        say(f"[{test_person}] epoch {epoch:>2}/{cfg.epochs}  "
            f"loss {running / n_batch:.4f}  val {val_err:.3f}°  "
            f"test {test_err:.3f}°"
            f"{' *' if best['epoch'] == epoch else ''}  "
            f"({(time.perf_counter() - t0) / 60:.1f}분)")

    # 검증 오차가 가장 낮았던 시점으로 되돌려 테스트를 잰다
    net.load_state_dict(best["state"])
    if track_test:
        pred = predict_all(net, test, norms)
        test_err = mean_angular_error(pred, test_cpu.gaze)
        say(f"[{test_person}] 완료: epoch {best['epoch']} 선택 → 테스트 {test_err:.3f}°")
    else:
        pred, test_err = None, float("nan")
        say(f"[{test_person}] 완료: epoch {best['epoch']} 선택 (검증 {best['val']:.3f}°)")

    # 나중에 다시 학습하지 않고 오차를 뜯어볼 수 있도록 예측과 가중치를 남긴다
    if fold_dir and pred is not None:
        os.makedirs(fold_dir, exist_ok=True)
        np.savez_compressed(os.path.join(fold_dir, f"{test_person}.npz"),
                            pred=pred, gaze=test_cpu.gaze, head=test_cpu.head,
                            **{f"w.{k}": v for k, v in net.state_dict().items()})
    if log_file:
        log_file.close()

    return {
        "test_person": test_person,
        "val_persons": val_persons,
        "test_error": test_err,
        "val_error": best["val"],
        "best_epoch": best["epoch"],
        "final_test_error": history[-1]["test"],
        "history": history,
        "seconds": time.perf_counter() - t0,
    }


def _worker(args):
    cfg, person, log_dir, fold_dir = args
    result = train_fold(cfg, person, log_dir=log_dir, fold_dir=fold_dir, verbose=False)
    print(f"[{person}] {result['test_error']:.3f}°  "
          f"(best epoch {result['best_epoch']}, val {result['val_error']:.3f}°, "
          f"{result['seconds'] / 60:.1f}분)", flush=True)
    return result


def run(cfg: Config, folds: list[str], jobs: int, log_dir: str, fold_dir: str) -> list[dict]:
    if jobs == 1:
        return [train_fold(cfg, p, log_dir=log_dir, fold_dir=fold_dir) for p in folds]

    from multiprocessing import get_context
    with get_context("spawn").Pool(jobs) as pool:
        return pool.map(_worker, [(cfg, p, log_dir, fold_dir) for p in folds])


def report(cfg: Config, results: list[dict], out_dir: str) -> None:
    from metrics import fold_table, fold_table_header, summary

    os.makedirs(out_dir, exist_ok=True)
    errors = {r["test_person"]: r["test_error"] for r in results}

    print("\n" + fold_table_header())
    print(fold_table("GazeNet (numpy)", errors))
    print("\n" + summary(errors))

    _collect_predictions(results, os.path.join(out_dir, "folds"),
                         os.path.join(out_dir, "predictions.npz"))

    path = os.path.join(out_dir, "lopo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2,
                  ensure_ascii=False)
    print(f"\n→ {path}")

    _plot(results, os.path.join(out_dir, "curves.png"))


def _collect_predictions(results, fold_dir: str, path: str) -> None:
    """fold별 npz에서 예측만 모아 가벼운 한 파일로. 가중치는 두고 온다."""
    if not os.path.isdir(fold_dir):
        return
    persons, pred, gaze, head = [], [], [], []
    for r in sorted(results, key=lambda r: r["test_person"]):
        f = os.path.join(fold_dir, f"{r['test_person']}.npz")
        if not os.path.exists(f):
            return
        with np.load(f) as z:
            pred.append(z["pred"]); gaze.append(z["gaze"]); head.append(z["head"])
        persons += [r["test_person"]] * len(pred[-1])
    np.savez_compressed(path, person=np.array(persons), pred=np.concatenate(pred),
                        gaze=np.concatenate(gaze), head=np.concatenate(head))
    print(f"→ {path}")


def _plot(results: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eda import _use_korean_font
    _use_korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=130)
    for r in sorted(results, key=lambda r: r["test_person"]):
        epochs = [h["epoch"] for h in r["history"]]
        axes[0].plot(epochs, [h["val"] for h in r["history"]], lw=0.8)
        axes[1].plot(epochs, [h["test"] for h in r["history"]], lw=0.8,
                     label=r["test_person"])
    axes[0].set(xlabel="epoch", ylabel="각도 오차 (°)", title="안쪽 검증 2명")
    axes[1].set(xlabel="epoch", title="테스트 1명 (선택에 쓰지 않음)")
    axes[1].legend(fontsize=5, ncol=3)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {path}")


def main() -> None:
    from dataset import PERSONS

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--folds", nargs="*", default=list(PERSONS), help="기본값은 15 fold 전부")
    p.add_argument("--jobs", type=int, default=15, help="동시에 돌릴 fold 수")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--dropout", type=float, default=Config.dropout)
    p.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    p.add_argument("--augment", action="store_true", help="shift·밝기·cutout 증강")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="gpu는 cupy로 같은 계층 코드를 GPU에서 돌린다")
    p.add_argument("--out", default="results/base")
    args = p.parse_args()

    cfg = Config(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 dropout=args.dropout, weight_decay=args.weight_decay,
                 augment=args.augment)
    jobs = min(args.jobs, len(args.folds))

    from common.backend import describe
    print(f"fold {len(args.folds)}개 · worker {jobs}개 · {describe()}")
    if args.device == "cpu":
        print(f"worker당 BLAS 스레드 {THREADS_PER_FOLD}개 (환경변수 GAZE_THREADS)")
    print(f"{cfg}")
    log_dir = os.path.join(args.out, "logs")
    fold_dir = os.path.join(args.out, "folds")
    print(f"fold별 진행 상황: {log_dir}/pXX.log")
    print(f"fold별 가중치·예측: {fold_dir}/pXX.npz\n")
    t0 = time.perf_counter()
    results = run(cfg, args.folds, jobs, log_dir, fold_dir)
    print(f"\n전체 {(time.perf_counter() - t0) / 60:.1f}분")
    report(cfg, results, args.out)


if __name__ == "__main__":
    main()
