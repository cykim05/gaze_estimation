#!/bin/bash
# 설정 탐색 — GPU 1장. 개발용 fold의 안쪽 검증 오차로만 고르고 테스트는 계산하지 않는다.
#
#   sbatch slurm/search.sbatch --epochs 20
#   sbatch slurm/search.sbatch --preset capacity --epochs 20
#SBATCH --job-name=gaze-search
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

echo "노드 $(hostname) · GPU ${CUDA_VISIBLE_DEVICES:-?} · 작업 ${SLURM_JOB_ID:-local}"
srun "$PYTHON" -u search.py --device gpu "$@"
