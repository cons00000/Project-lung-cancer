#!/bin/bash
#SBATCH --job-name=setup_lung_env
#SBATCH --output=setup_lung_env_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_inter

set -e  # Stoppe le script si une commande échoue

# Initialiser conda
source $(conda info --base)/etc/profile.d/conda.sh

# Créer l'environnement s'il n'existe pas
if ! conda info --envs | grep -q '^lung_env'; then
    conda create -n lung_env python=3.10 -y
fi

# Activer l'environnement
conda activate lung_env

# Installer les packages conda avec versions compatibles
conda install -y -c conda-forge \
  keras=2.13.1 \
  opencv=4.6.0 \
  scikit-image=0.20.0 \
  scipy=1.10.1 \
  tqdm=4.66.1 \
  statsmodels=0.14.1 \
  matplotlib=3.7.1 \
  simpleitk=2.2.1 \
  ipykernel=6.25.0 \
  numpy=1.23.5 \
  ipython=8.13 \
  shap

# Installer les packages pip avec versions compatibles
pip install \
  tensorflow==2.13.0 \
  h5py==3.9.0 \
  typing_extensions==4.5.0 \
  ipython==8.12.2 \
  papermill==2.4.0 \
  xplique


# Ajouter l’environnement comme noyau Jupyter
python -m ipykernel install --user --name=lung_env --display-name "Python (lung_env)"

echo "✅ Environnement conda 'lung_env' prêt avec kernel Jupyter 'Python (lung_env)'."
