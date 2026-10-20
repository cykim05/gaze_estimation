#!/bin/bash
# Phase 2 — 15-fold 실행을 작업 배열로. %2로 동시 GPU 2장 상한을 건다.
#   sbatch --array=0-9%2 slurm/phase2.sbatch
#SBATCH --job-name=gaze-p2
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs
export PYTHONIOENCODING=utf-8
export LANG=${LANG:-C.UTF-8}
export GAZE_GPU_FRACTION=${GAZE_GPU_FRACTION:-0.30}

LIST=${GAZE_LIST:-slurm/phase2_configs.txt}
# 주석과 빈 줄을 뺀 n번째 줄
LINE=$(grep -vE '^\s*(#|$)' "$LIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
NAME=$(echo "$LINE" | awk '{print $1}')
ARGS=$(echo "$LINE" | cut -d' ' -f2-)

echo "노드 $(hostname) · GPU ${CUDA_VISIBLE_DEVICES:-?} · 작업 ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "이름 $NAME · 인자 $ARGS"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

srun /data/hai_chanyoung/miniconda3/envs/gaze/bin/python -u train.py \
  --device gpu --jobs 2 --per-gpu 2 --min-free 2 \
  --epochs 20 --dropout 0.5 --batch-size 128 \
  --out "results/$NAME" $ARGS
