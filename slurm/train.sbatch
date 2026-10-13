#!/bin/bash
# leave-one-person-out 15 fold 학습 — GPU 1장.
#
#   sbatch slurm/train.sbatch                                   # 기본 설정
#   sbatch slurm/train.sbatch --epochs 20 --augment --dropout 0.5 --weight-decay 1e-4
#
# 인자는 그대로 train.py로 넘어간다. --device gpu와 --out은 아래에서 붙인다.
#SBATCH --job-name=gaze-train
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs

PYTHON=/data/hai_chanyoung/miniconda3/envs/gaze/bin/python
OUT=${GAZE_OUT:-results/gpu}

echo "노드 $(hostname) · GPU ${CUDA_VISIBLE_DEVICES:-?} · 작업 ${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# fold는 순차로 돈다. GPU 1장에서 프로세스를 늘려도 커널 실행에 묶여 이득이 적고,
# 메모리는 fold당 CUDA 문맥만큼 더 든다.
srun "$PYTHON" -u train.py --device gpu --jobs "${GAZE_JOBS:-1}" --out "$OUT" "$@"
