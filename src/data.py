import os
import pandas as pd
from monai.transforms import LoadImage, Compose, EnsureChannelFirst, ScaleIntensity
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List

## données sous forme d'un dataframe (contient aussi les labels)
clinical_data = pd.read_excel('/Users/constance/Documents/Project_lung_cancer/NIH dataset_raw/statistics-clinical-20201221.xlsx')

# afficher les statistiques descriptives du dataframe
print(clinical_data.describe(include='all')) # les colonnes sont: No. ; NewPatientID ; Sex ; Age ; weight (kg) ; T-Stage ; N-Stage ; Ｍ-Stage ; Histopathological grading ; Smoking History                0

# afficher le nombre de données manquantes dans chaque colonne
print(clinical_data.isnull().sum())

@dataclass
class Patient:
    No: int
    PatientID: str
    Sex: str
    Age: int
    Weight: float
    SmokingHistory: int
    images: List[str]  # Liste des chemins des images associées à ce patient

    def __str__(self):
        return (
            f"Patient No. {self.No}, ID: {self.PatientID}, "
            f"Sex: {self.Sex}, Age: {self.Age}, Weight: {self.Weight}kg, "
            f"Smoking History: {self.SmokingHistory}, "
            f"Images: {len(self.images)}"
        )

    def add_image(self, image_path: str):
        """Ajoute une image à la liste des images du patient"""
        self.images.append(image_path)

# Calculer la moyenne de l'âge et du poids en ignorant les NaN
mean_age = clinical_data['Age'].mean()
mean_weight = clinical_data['weight (kg)'].mean()

# Remplacer les NaN par la moyenne dans les colonnes correspondantes
clinical_data['Age'] = clinical_data['Age'].fillna(mean_age)
clinical_data['weight (kg)'] = clinical_data['weight (kg)'].fillna(mean_weight)

patients = {
    row['NewPatientID']: Patient(
        No=row['No.'],
        PatientID=row['NewPatientID'],
        Sex=row['Sex'],
        Age=int(row['Age']),
        Weight=float(row['weight (kg)']),
        SmokingHistory=row['Smoking History'],
        images=[]  # Liste des images pour chaque patient
    )
    for _, row in clinical_data.iterrows()
}

# charger les données dicom

# Étape 1 — récupère les chemins valides
def get_valid_image_paths(root_dir, file_extension='.dcm'):
    valid_paths = []
    loader = LoadImage()

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(file_extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    loader(file_path)  # test de chargement uniquement
                    valid_paths.append(file_path)
                except Exception:
                    pass
    return valid_paths

# Étape 2 — dataset personnalisé
class DicomDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform or Compose([
            LoadImage(),
            EnsureChannelFirst(),        
            ScaleIntensity()              
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = self.transform(img_path)
        return image  

# Étape 3 — utilisation dans un DataLoader
root_directory = '/Users/constance/Documents/Project_lung_cancer/NIH dataset_raw/manifest-1608669183333/Lung-PET-CT-Dx'
# valid_paths = get_valid_image_paths(root_directory)

# dataset = DicomDataset(valid_paths)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
