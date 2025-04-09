import pandas as pd



## données sous forme d'un dataframe (contient aussi les labels)
clinical_data = pd.read_excel('/Users/constance/Documents/Project_lung_cancer/NIH dataset_raw/statistics-clinical-20201221.xlsx')

# afficher les statistiques descriptives du dataframe
print(clinical_data.describe(include='all')) # les colonnes sont: No. ; NewPatientID ; Sex ; Age ; weight (kg) ; T-Stage ; N-Stage ; Ｍ-Stage ; Histopathological grading ; Smoking History                0

# afficher le nombre de données manquantes dans chaque colonne
print(clinical_data.isnull().sum())




## créer une classe pour les patients
class Patients:
    def __init__(self, No, PatientID, Sex, Age, Weight, Smocking_history):

        self.No = No
        self.PatientID = PatientID
        self.sex=Sex
        self.age=Age
        self.weight=Weight
        self.smoking_history=Smocking_history

    def __str__(self):
        return f"No. {self.No}, ID {self.PatientID}, Sex {self.sex}, Age {self.age}, Weight {self.weight}, Smoking history {self.smoking_history}"

patients = [Patients(row['No.'], row['NewPatientID'], row['Sex'],row['Age'], row['weight (kg)'], row['Smoking History']) for index, row in clinical_data.iterrows()]



