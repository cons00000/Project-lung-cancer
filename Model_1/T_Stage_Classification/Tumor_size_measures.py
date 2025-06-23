import sys
sys.path.append('/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Segmentation')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label
from joblib import Parallel, delayed
import recist_and_volume_calculator as rc
import time
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# Configuration des chemins
BASE_PATH = '/mounts/Datasets4/pred_lung_cancer/NIH dataset_raw/Processed'
CACHE_DIR = './data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache des masques
def load_cached_mask(patient_dir):
    patient_id = os.path.basename(patient_dir)
    cache_path = os.path.join(CACHE_DIR, f"{patient_id}_mask.npy")
    
    if os.path.exists(cache_path):
        return np.load(cache_path), patient_id
    
    mask_path = os.path.join(patient_dir, 'DL_mask.nrrd')
    if not os.path.exists(mask_path):
        return None, patient_id
    
    try:
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        np.save(cache_path, mask)
        return mask, patient_id
    except Exception:
        return None, patient_id

# Méthode alternative pour calculer le volume (formule ellipsoïde)
def calculate_ellipsoid_volume(mask, spacing):
    """Calcule le volume en utilisant la formule d'ellipsoïde basée sur les axes principaux"""
    # Obtenir les coordonnées des voxels de la tumeur
    z, y, x = np.where(mask > 0)
    if len(x) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Convertir en coordonnées physiques (mm)
    x_mm = x * spacing[0]
    y_mm = y * spacing[1]
    z_mm = z * spacing[2]
    points = np.vstack((x_mm, y_mm, z_mm)).T
    
    # Calculer la matrice de covariance
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov_matrix = np.cov(centered.T)
    
    # Calculer les valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Trier par ordre décroissant
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    
    # Calculer les diamètres (longueurs des axes principaux)
    diameters = 4 * np.sqrt(eigenvalues)  # 2 * demi-axe
    
    # Volume de l'ellipsoïde: V = 4/3 * π * a * b * c
    volume = (4/3) * np.pi * (diameters[0]/2) * (diameters[1]/2) * (diameters[2]/2)
    
    # Les trois diamètres principaux
    major_diam = diameters[0]  # Diamètre le plus long
    minor_diam = diameters[1]  # Diamètre intermédiaire
    
    return volume, diameters[0], diameters[1], diameters[2]

# Méthode alternative pour calculer le RECIST (diamètre maximal dans le plan axial)
def calculate_max_diameter(mask, spacing):
    """Calcule le diamètre maximal dans la slice avec la plus grande aire"""
    # Trouver la slice avec la plus grande aire tumorale
    slice_areas = np.sum(mask, axis=(1, 2))
    if np.sum(slice_areas) == 0:
        return 0.0, 0
    
    slice_idx = np.argmax(slice_areas)
    
    # Extraire le masque 2D pour cette slice
    mask_2d = mask[slice_idx]
    y, x = np.where(mask_2d > 0)
    
    if len(x) == 0:
        return 0.0, slice_idx
    
    # Convertir en coordonnées physiques (mm)
    x_mm = x * spacing[0]
    y_mm = y * spacing[1]
    points = np.vstack((x_mm, y_mm)).T
    
    # Calculer l'enveloppe convexe
    if len(points) < 3:
        # Pour moins de 3 points, utiliser la distance maximale directement
        distances = cdist(points, points)
        max_diameter = np.max(distances)
        return max_diameter, slice_idx
    
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # Calculer toutes les distances entre les points de l'enveloppe convexe
        distances = cdist(hull_points, hull_points)
        
        # Trouver la distance maximale
        max_diameter = np.max(distances)
        
        return max_diameter, slice_idx
    except:
        # Fallback si l'enveloppe convexe échoue
        distances = cdist(points, points)
        max_diameter = np.max(distances)
        return max_diameter, slice_idx

# Traitement d'un patient
def process_patient(patient_dir):
    mask_array, patient_id = load_cached_mask(patient_dir)
    if mask_array is None:
        return None, None
    
    try:
        # Estimation de l'espacement
        img_path = os.path.join(patient_dir, 'image.nrrd')
        spacing = (1.0, 1.0, 1.0)
        if os.path.exists(img_path):
            img = sitk.ReadImage(img_path)
            spacing = img.GetSpacing()
            # Vérification de l'unité d'espacement
            if max(spacing) > 10:  # Si l'espacement est en mm mais trop grand
                print(f"Attention: espacement suspect pour {patient_id}: {spacing}")
        
        # Segmentation des tumeurs
        structure = np.ones((3, 3, 3), dtype=int)
        labeled_mask, num_nodules = label(mask_array, structure=structure)
        
        # Variables pour les résultats
        tumor_data = []
        max_recist_A = 0
        max_recist_B = 0
        
        for tumor_id in range(1, num_nodules + 1):
            tumor_mask = (labeled_mask == tumor_id).astype(np.uint8)
            
            # Méthode A: Calcul original avec rc.calculate_values()
            try:
                recist_pred_A, volume_pred_A, slice_idx_A, _ = rc.calculate_values(tumor_mask, spacing)
                # Conversion probable de cm³ en mm³ (1 cm³ = 1000 mm³)
                volume_pred_A *= 1000
            except Exception as e:
                print(f"Erreur dans rc.calculate_values pour {patient_id}-t{tumor_id}: {str(e)}")
                recist_pred_A, volume_pred_A, slice_idx_A = 0.0, 0.0, 0
            
            # Méthode B: Calcul alternatif
            try:
                volume_pred_B, diam1, diam2, diam3 = calculate_ellipsoid_volume(tumor_mask, spacing)
                recist_pred_B, slice_idx_B = calculate_max_diameter(tumor_mask, spacing)
            except Exception as e:
                print(f"Erreur dans méthode B pour {patient_id}-t{tumor_id}: {str(e)}")
                volume_pred_B, diam1, diam2, diam3, recist_pred_B, slice_idx_B = 0.0, 0.0, 0.0, 0.0, 0.0, 0
            
            # Mise à jour des RECIST maximaux
            if recist_pred_A > max_recist_A: 
                max_recist_A = recist_pred_A
            if recist_pred_B > max_recist_B: 
                max_recist_B = recist_pred_B
            
            tumor_data.append({
                'patient_id': patient_id,
                'tumor_id': tumor_id,
                # Méthode A (rc.calculate_values)
                'recist_mm_A': recist_pred_A,
                'volume_mm3_A': volume_pred_A,
                'representative_slice_A': slice_idx_A,
                # Méthode B (alternative)
                'recist_mm_B': recist_pred_B,
                'volume_mm3_B': volume_pred_B,
                'diameter1_mm_B': diam1,
                'diameter2_mm_B': diam2,
                'diameter3_mm_B': diam3,
                'representative_slice_B': slice_idx_B,
                # Informations supplémentaires
                'spacing_x': spacing[0],
                'spacing_y': spacing[1],
                'spacing_z': spacing[2],
                'voxel_count': np.sum(tumor_mask)
            })
        
        return tumor_data, {
            'patient_id': patient_id,
            'num_nodules': num_nodules,
            'max_recist_mm_A': max_recist_A,
            'max_recist_mm_B': max_recist_B,
            'mean_volume_mm3_A': np.mean([t['volume_mm3_A'] for t in tumor_data]),
            'mean_volume_mm3_B': np.mean([t['volume_mm3_B'] for t in tumor_data])
        }
        
    except Exception as e:
        print(f"Erreur globale pour {patient_id}: {str(e)}")
        return None, None

# Pipeline principal
if __name__ == "__main__":
    start_time = time.time()
    
    # Liste des patients
    patients = [
        os.path.join(BASE_PATH, d) 
        for d in os.listdir(BASE_PATH) 
        if os.path.isdir(os.path.join(BASE_PATH, d))
    ]
    print(f"Traitement de {len(patients)} patients...")
    
    # Traitement parallèle
    n_jobs = max(1, os.cpu_count() - 2)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_patient)(patient_dir) 
        for patient_dir in patients
    )
    
    # Agrégation des résultats
    all_tumors = []
    all_summaries = []
    
    for tumor_data, summary in results:
        if tumor_data and summary:
            all_tumors.extend(tumor_data)
            all_summaries.append(summary)
    
    # Sauvegarde des résultats
    if all_tumors:
        tumor_df = pd.DataFrame(all_tumors)
        summary_df = pd.DataFrame(all_summaries)
        
        # Chemins de sortie
        CSV_TUMOR_PATH = 'Model_1/T_Stage_Classification/tumor_volume_analysis_comparison.csv'
        CSV_SUMMARY_PATH = 'Model_1/T_Stage_Classification/patient_volume_summaries_comparison.csv'
        
        # Sauvegarde CSV
        tumor_df.to_csv(CSV_TUMOR_PATH, index=False)
        summary_df.to_csv(CSV_SUMMARY_PATH, index=False)
        
        # Rapport final
        duration = time.time() - start_time
        report = f"""
        RAPPORT D'ANALYSE VOLUMÉTRIQUE COMPARATIVE
        ===========================================
        Patients traités: {len(summary_df)}
        Tumeurs analysées: {len(tumor_df)}
        Temps total: {duration:.1f} secondes
        Temps moyen par patient: {duration/max(1, len(summary_df)):.2f} sec
        
        STATISTIQUES CLÉS (MÉTHODE A):
        - RECIST maximal moyen: {summary_df['max_recist_mm_A'].mean():.1f} mm
        - Volume tumoral moyen: {tumor_df['volume_mm3_A'].mean():.1f} mm³
        - Volume médian: {tumor_df['volume_mm3_A'].median():.1f} mm³
        
        STATISTIQUES CLÉS (MÉTHODE B):
        - RECIST maximal moyen: {summary_df['max_recist_mm_B'].mean():.1f} mm
        - Volume tumoral moyen: {tumor_df['volume_mm3_B'].mean():.1f} mm³
        - Volume médian: {tumor_df['volume_mm3_B'].median():.1f} mm³
        
        CORRÉLATION RECIST/VOLUME (MÉTHODE A):
        - Corrélation: {tumor_df['recist_mm_A'].corr(tumor_df['volume_mm3_A']):.3f}
        
        CORRÉLATION RECIST/VOLUME (MÉTHODE B):
        - Corrélation: {tumor_df['recist_mm_B'].corr(tumor_df['volume_mm3_B']):.3f}
        
        CORRÉLATION INTER-MÉTHODES:
        - RECIST: {tumor_df['recist_mm_A'].corr(tumor_df['recist_mm_B']):.3f}
        - Volume: {tumor_df['volume_mm3_A'].corr(tumor_df['volume_mm3_B']):.3f}
        
        DISTRIBUTION DES VOLUMES (MÉTHODE A):
        - Min: {tumor_df['volume_mm3_A'].min():.1f} mm³
        - Max: {tumor_df['volume_mm3_A'].max():.1f} mm³
        - Écart-type: {tumor_df['volume_mm3_A'].std():.1f} mm³
        
        DISTRIBUTION DES VOLUMES (MÉTHODE B):
        - Min: {tumor_df['volume_mm3_B'].min():.1f} mm³
        - Max: {tumor_df['volume_mm3_B'].max():.1f} mm³
        - Écart-type: {tumor_df['volume_mm3_B'].std():.1f} mm³
        
        DISTRIBUTION DES NODULES:
        {summary_df['num_nodules'].value_counts().sort_index()}
        """
        print(report)
        
        with open('volume_comparison_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Résultats sauvegardés dans:")
        print(f"- Détails tumeurs: {CSV_TUMOR_PATH}")
        print(f"- Résumés patients: {CSV_SUMMARY_PATH}")
        
        # Analyse supplémentaire
        print("\nANALYSE RECIST vs VOLUME (Méthode A):")
        print(tumor_df[['recist_mm_A', 'volume_mm3_A']].describe())
        
        print("\nANALYSE RECIST vs VOLUME (Méthode B):")
        print(tumor_df[['recist_mm_B', 'volume_mm3_B']].describe())
        
        # Option: Visualisation des résultats

        import matplotlib.pyplot as plt
        
        # Comparaison des RECIST
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.scatter(tumor_df['recist_mm_A'], tumor_df['recist_mm_B'], alpha=0.5)
        max_recist = max(tumor_df['recist_mm_A'].max(), tumor_df['recist_mm_B'].max())
        plt.plot([0, max_recist], [0, max_recist], 'r--')
        plt.title('Comparaison des mesures RECIST')
        plt.xlabel('RECIST Méthode A (mm)')
        plt.ylabel('RECIST Méthode B (mm)')
        plt.grid(True)
        
        # Comparaison des Volumes
        plt.subplot(122)
        plt.scatter(tumor_df['volume_mm3_A'], tumor_df['volume_mm3_B'], alpha=0.5)
        max_vol = max(tumor_df['volume_mm3_A'].max(), tumor_df['volume_mm3_B'].max())
        plt.plot([0, max_vol], [0, max_vol], 'r--')
        plt.title('Comparaison des volumes tumoraux')
        plt.xlabel('Volume Méthode A (mm³)')
        plt.ylabel('Volume Méthode B (mm³)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png')
        plt.show()
        
        print("Graphique sauvegardé: method_comparison.png")
        
    else:
        print("Aucune donnée valide à exporter")