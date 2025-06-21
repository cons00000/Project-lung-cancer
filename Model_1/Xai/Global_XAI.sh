#!/bin/bash
#SBATCH --job-name=global_XAI_job
#SBATCH --output=global_XAI_%j.out
#SBATCH --error=global_XAI_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_prod_long
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

PYTHON_SCRIPT="/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Xai/Global_XAI.py"

# Exécution avec logs
echo "### Début de l'exécution : $(date)"
python "$PYTHON_SCRIPT"
echo "### Fin de l'exécution : $(date)"