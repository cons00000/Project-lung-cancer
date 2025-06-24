#!/bin/bash
#SBATCH --job-name=measure_job
#SBATCH --output=measure_%j.out
#SBATCH --error=measure_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_prod_long
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

PYTHON_SCRIPT="/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/T_Stage_Classification/Tumor_size_measures.py"

echo "### Start of the execution : $(date)"
python "$PYTHON_SCRIPT"
echo "### End of the exécution : $(date)"