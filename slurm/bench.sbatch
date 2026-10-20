#!/bin/bash
#SBATCH --job-name=gaze-b0
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs
export PYTHONIOENCODING=utf-8
export LANG=${LANG:-C.UTF-8}
echo "노드 $(hostname) · GPU ${CUDA_VISIBLE_DEVICES:-?} · 작업 ${SLURM_JOB_ID:-local}"
srun /data/hai_chanyoung/miniconda3/envs/gaze/bin/python -u bench_throughput.py --device gpu "$@"
