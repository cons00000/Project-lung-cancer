#!/bin/bash
#SBATCH --job-name=seg_job
#SBATCH --output=seg_%j.out
#SBATCH --error=seg_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_prod_long
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate lung_env

NOTEBOOK_PATH=/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/src/Segmentation/Segmentation.ipynb
OUTPUT_NOTEBOOK=${NOTEBOOK_PATH%.ipynb}_output.ipynb

# Exécution avec logs
echo "### Début de l'exécution : $(date)"
papermill "$NOTEBOOK_PATH" "$OUTPUT_NOTEBOOK" --log-output
echo "### Fin de l'exécution : $(date)"

