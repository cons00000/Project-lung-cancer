import os
import pandas as pd
from collections import Counter
from monai.transforms import LoadImage, Compose, EnsureChannelFirst, ScaleIntensity
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List
import xml.etree.ElementTree as ET

#####################################
# Définition des classes et fonctions
#####################################

# Chargement des données cliniques

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


# Chargement des images DICOM

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
   

# Chargement des annotations XML

def count_files_with_extension(root_dir, extension):
    count = 0
    for _, _, files in os.walk(root_dir):
        count += len([f for f in files if f.lower().endswith(extension)])
    return count

def verify_annotations_vs_dicoms():
    all_patients = sorted(os.listdir(ANNOTATION_DIR))
    for patient_id in all_patients:
        xml_dir = os.path.join(ANNOTATION_DIR, patient_id)
        dicom_dir = os.path.join(DICOM_ROOT_DIR, f"Lung_Dx-{patient_id}")

        if not os.path.isdir(xml_dir):
            print(f"❌ Dossier XML manquant pour {patient_id}")
            continue

        if not os.path.isdir(dicom_dir):
            print(f"❌ Dossier DICOM manquant pour {patient_id}")
            continue

        xml_count = count_files_with_extension(xml_dir, ".xml")
        dcm_count = count_files_with_extension(dicom_dir, ".dcm")

        status = "✅ OK" if xml_count == dcm_count else "⚠️ Mismatch"
        print(f"{status} | {patient_id} → XML: {xml_count} / DICOM: {dcm_count}")

def find_first_dicom(patient_id):
    """
    Cherche le premier fichier .dcm pour un patient donné
    dans son dossier Lung_Dx-XXX
    """
    patient_folder = os.path.join(DICOM_ROOT_DIR, f"Lung_Dx-{patient_id}")
    for dirpath, _, filenames in os.walk(patient_folder):
        for f in filenames:
            if f.lower().endswith(".dcm"):
                return os.path.join(dirpath, f)
    return None

def update_xml(xml_path):
    # Parse le fichier XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Récupère l'ID patient à partir du chemin du dossier parent
    patient_id = os.path.basename(os.path.dirname(xml_path))
    dicom_path = find_first_dicom(patient_id)

    if dicom_path is None:
        print(f"❌ Aucun fichier DICOM trouvé pour {patient_id}")
        return

    # Extraction des infos DICOM
    dicom_filename = os.path.basename(dicom_path)
    dicom_folder = os.path.basename(os.path.dirname(dicom_path))

    # Mise à jour des balises existantes
    for tag in ['folder', 'filename', 'path']:
        elem = root.find(tag)
        if elem is not None:
            elem.text = dicom_folder if tag == 'folder' else dicom_filename if tag == 'filename' else dicom_path

    # Gestion de la balise source/database
    source = root.find('source')
    if source is None:
        source = ET.SubElement(root, 'source')
    
    database = source.find('database')
    if database is None:
        database = ET.SubElement(source, 'database')
    database.text = "NIH_Lung_Cancer_Dataset"

    # Sauvegarde avec indentation pour un meilleur formatage
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"✅ Fichier XML mis à jour : {xml_path}")
    
def process_all_annotations(annotation_root):
    for dirpath, _, filenames in os.walk(annotation_root):
        for file in filenames:
            if file.endswith(".xml"):
                update_xml(os.path.join(dirpath, file))

########################
# Chargement des données
########################
clinical_data = pd.read_excel('/Users/constance/Documents/Project_lung_cancer/NIH dataset_raw/statistics-clinical-20201221.xlsx')


# Calculer la moyenne de l'âge et du poids en ignorant les NaN

mean_age = clinical_data['Age'].mean()
mean_weight = clinical_data['weight (kg)'].mean()


# Remplacer les NaN par la moyenne dans les colonnes correspondantes

clinical_data['Age'] = clinical_data['Age'].fillna(mean_age)
clinical_data['weight (kg)'] = clinical_data['weight (kg)'].fillna(mean_weight)

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

root_directory = '/Users/constance/Documents/Project_lung_cancer/NIH dataset_raw/manifest-1608669183333/Lung-PET-CT-Dx'
# valid_paths = get_valid_image_paths(root_directory)

# dataset = DicomDataset(valid_paths)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Dossiers racine pour établir la correspondance entre les fichiers DICOM et XML

ANNOTATION_DIR = "/Users/constance/Documents/Project_lung_cancer/NIH dataset_raw/Annotation"
DICOM_ROOT_DIR = "/Users/constance/Documents/Project_lung_cancer/NIH dataset_raw/manifest-1608669183333/Lung-PET-CT-Dx"


# Lancer la vérification
verify_annotations_vs_dicoms()


# Lancer la mise à jour
# process_all_annotations(ANNOTATION_DIR)