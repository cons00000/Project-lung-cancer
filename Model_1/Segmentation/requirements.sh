# Création ou activation d'un environnement conda avec Python 3.10
conda create -n mon_env python=3.10 -y
conda activate mon_env

# Installation des packages via conda
conda install -y \
  keras=2.13.1 \
  opencv=4.6.0 \
  scikit-image=0.21.0 \
  scipy=1.10.1 \
  tqdm=4.66.1 \
  statsmodels=0.14.1 \
  matplotlib=3.7.1 \
  simpleitk \
  pynrrd \
  ipykernel \
  openpyxl \

# Installation des packages via pip
pip install \
  tensorflow==2.13.0 \
  h5py==3.13.0 \
  typing_extensions==4.7.1
