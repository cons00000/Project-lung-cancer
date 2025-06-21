#!/bin/bash
#SBATCH --job-name=gradcam_job
#SBATCH --output=gradcam_%j.out
#SBATCH --error=gradcam_%j.err
#SBATCH --time=02:00:00            # 1 heure (ajuste selon ton besoin)
#SBATCH --partition=gpu_inter            # ou la partition adaptée
#SBATCH --mem=16G                  # mémoire (ajuste)
#SBATCH --cpus-per-task=4          # CPU (ajuste)

# Charge conda et active ton env
conda init
conda activate lung_env

# Chemin vers ton notebook sur le cluster
NOTEBOOK_PATH=/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Xai/Local_XAI.ipynb

# Lancer l’exécution du notebook avec papermill
papermill $NOTEBOOK_PATH ${NOTEBOOK_PATH%.ipynb}_output.ipynb
