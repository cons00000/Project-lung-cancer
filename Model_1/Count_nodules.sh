#!/bin/bash
#SBATCH --job-name=nodules_job
#SBATCH --output=Model_1/nodules/nodules_%j.out
#SBATCH --error=Model_1/nodules/nodules_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_prod_long
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

PYTHON_SCRIPT="/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Count_nodules.py"

# Exécution avec logs
echo "### Début de l'exécution : $(date)"
python "$PYTHON_SCRIPT"
echo "### Fin de l'exécution : $(date)"