#!/bin/bash
#SBATCH --job-name=hello-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1g
#SBATCH --time=00:02:00
#SBATCH --partition=ghx4
#SBATCH --account=bgjw-dtai-gh
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "Hello from Slurm!"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Running on host: $(hostname)"
echo "Current time: $(date)"
