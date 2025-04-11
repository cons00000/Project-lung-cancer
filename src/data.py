import os
import pandas as pd
from collections import Counter
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

class Diagnostic:
    def __init__(self, patient_id: str, t_stage: str, n_stage: str, m_stage: str, hispastological_grading: str):
        self.patient_id = patient_id
        self.t_stage = t_stage
        self.n_stage = n_stage
        self.m_stage = m_stage
        self.hispastological_grading = hispastological_grading

    def __str__(self):
        return (
            f"Patient ID: {self.patient_id}, T-Stage: {self.t_stage}, "
            f"N-Stage: {self.n_stage}, M-Stage: {self.m_stage}"
            f"Histopathological Grading: {self.hispastological_grading}"
        )
# Calculer la moyenne de l'âge et du poids en ignorant les NaN
mean_age = clinical_data['Age'].mean()
mean_weight = clinical_data['weight (kg)'].mean()

# Remplacer les NaN par la moyenne dans les colonnes correspondantes
clinical_data['Age'] = clinical_data['Age'].fillna(mean_age)
clinical_data['weight (kg)'] = clinical_data['weight (kg)'].fillna(mean_weight)

diagnostic = {
    row['NewPatientID']: Diagnostic(
        patient_id=row['NewPatientID'],
        t_stage=row['T-Stage'],
        n_stage=row['N-Stage'],
        m_stage=row['Ｍ-Stage'],
        hispastological_grading=row['Histopathological grading']
    )
    for _, row in clinical_data.iterrows()
}


# Compter les occurrences pour chaque champ
t_stage_counts = Counter(diag.t_stage for diag in diagnostic.values())
n_stage_counts = Counter(diag.n_stage for diag in diagnostic.values())
m_stage_counts = Counter(diag.m_stage for diag in diagnostic.values())
grading_counts = Counter(diag.hispastological_grading for diag in diagnostic.values())

# Affichage des résultats
print("Occurrences par valeur :")
print("T-Stage:")
for val, count in t_stage_counts.items():
    print(f"  {val}: {count}")

# is : 3
# 1: 2  1a: 9  1b: 29  1c: 127
# 2: 53 2a: 37  2b: 15 
# 3: 57
# 4: 23

print("N-Stage:")
for val, count in n_stage_counts.items():
    print(f"  {val}: {count}")

# 0: 184
# 1: 85
# 2: 8
# 3: 78

print("M-Stage:")
for val, count in m_stage_counts.items():
    print(f"  {val}: {count}")

# 0: 230
# 1: 53 1a: 30 1b: 26 1c: 13
# 2: 1
# 3: 2

print("Histopathological Grading:")
for val, count in grading_counts.items():
    print(f"  {val}: {count}")

# G1: 11  " G"1: 1 G1-2: 7 G1-G2: 2
# G2: 27  G2-3: 34  G2-G3: 1  
# G3: 61  " G3": 1

patients = {
    row['NewPatientID']: Patient(
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
