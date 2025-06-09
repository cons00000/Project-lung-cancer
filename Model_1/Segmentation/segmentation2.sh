#!/bin/bash
#SBATCH --job-name=seg_M1_job
#SBATCH --output=seg_M1_%j.out
#SBATCH --error=seg_M1_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_prod_long
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

PYTHON_SCRIPT="/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Segmentation/segmentation.py"

# Exécution avec logs
echo "### Début de l'exécution : $(date)"
python "$PYTHON_SCRIPT"
echo "### Fin de l'exécution : $(date)"