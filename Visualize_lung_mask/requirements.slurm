#!/bin/bash
#SBATCH --job-name=setup_vis_env
#SBATCH --output=setup_vis_env_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_inter

set -e  # Arrête le script en cas d’erreur

# Initialiser conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Créer l’environnement s’il n’existe pas
if ! conda info --envs | grep -q '^vis_env'; then
    conda create -n vis_env python=3.11 -y
fi

# Activer l’environnement
conda activate vis_env

# Mise à jour pip
python -m pip install --upgrade pip

# Installer les packages depuis conda-forge
conda install -y -c conda-forge \
  pynrrd \
  napari \
  magicgui \
  ipykernel

# Ajouter l’environnement comme noyau Jupyter
python -m ipykernel install --user --name=vis_env --display-name "Python (vis_env)"

echo "✅ Environnement 'vis_env' prêt avec les dernières versions de napari, pynrrd, etc."
