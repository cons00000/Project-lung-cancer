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

# Path configuration
BASE_PATH = '/mounts/Datasets4/pred_lung_cancer/NIH dataset_raw/Processed'
CACHE_DIR = './data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Mask caching
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

# Alternative method to calculate volume (ellipsoid formula)
def calculate_ellipsoid_volume(mask, spacing):
    """Calculate volume using ellipsoid formula based on principal axes"""
    # Get coordinates of tumor voxels
    z, y, x = np.where(mask > 0)
    if len(x) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Convert to physical coordinates (mm)
    x_mm = x * spacing[0]
    y_mm = y * spacing[1]
    z_mm = z * spacing[2]
    points = np.vstack((x_mm, y_mm, z_mm)).T
    
    # Calculate covariance matrix
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov_matrix = np.cov(centered.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    
    # Calculate diameters (principal axis lengths)
    diameters = 4 * np.sqrt(eigenvalues)  # 2 * semi-axis
    
    # Ellipsoid volume: V = 4/3 * π * a * b * c
    volume = (4/3) * np.pi * (diameters[0]/2) * (diameters[1]/2) * (diameters[2]/2)
    
    # Three principal diameters
    major_diam = diameters[0]  # Longest diameter
    minor_diam = diameters[1]  # Intermediate diameter
    
    return volume, diameters[0], diameters[1], diameters[2]

# Alternative method to calculate RECIST (maximum diameter in axial plane)
def calculate_max_diameter(mask, spacing):
    """Calculate maximum diameter in the slice with the largest area"""
    # Find slice with largest tumor area
    slice_areas = np.sum(mask, axis=(1, 2))
    if np.sum(slice_areas) == 0:
        return 0.0, 0
    
    slice_idx = np.argmax(slice_areas)
    
    # Extract 2D mask for this slice
    mask_2d = mask[slice_idx]
    y, x = np.where(mask_2d > 0)
    
    if len(x) == 0:
        return 0.0, slice_idx
    
    # Convert to physical coordinates (mm)
    x_mm = x * spacing[0]
    y_mm = y * spacing[1]
    points = np.vstack((x_mm, y_mm)).T
    
    # Calculate convex hull
    if len(points) < 3:
        # For less than 3 points, use maximum distance directly
        distances = cdist(points, points)
        max_diameter = np.max(distances)
        return max_diameter, slice_idx
    
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        # Calculate all distances between convex hull points
        distances = cdist(hull_points, hull_points)
        
        # Find maximum distance
        max_diameter = np.max(distances)
        
        return max_diameter, slice_idx
    except:
        # Fallback if convex hull fails
        distances = cdist(points, points)
        max_diameter = np.max(distances)
        return max_diameter, slice_idx

# Process a patient
def process_patient(patient_dir):
    mask_array, patient_id = load_cached_mask(patient_dir)
    if mask_array is None:
        return None, None
    
    try:
        # Spacing estimation
        img_path = os.path.join(patient_dir, 'image.nrrd')
        spacing = (1.0, 1.0, 1.0)
        if os.path.exists(img_path):
            img = sitk.ReadImage(img_path)
            spacing = img.GetSpacing()
            # Check spacing unit
            if max(spacing) > 10:  # If spacing is in mm but too large
                print(f"Warning: suspicious spacing for {patient_id}: {spacing}")
        
        # Tumor segmentation
        structure = np.ones((3, 3, 3), dtype=int)
        labeled_mask, num_nodules = label(mask_array, structure=structure)
        
        # Variables for results
        tumor_data = []
        max_recist_A = 0
        max_recist_B = 0
        
        for tumor_id in range(1, num_nodules + 1):
            tumor_mask = (labeled_mask == tumor_id).astype(np.uint8)
            
            # Method A: Original calculation with rc.calculate_values()
            try:
                recist_pred_A, volume_pred_A, slice_idx_A, _ = rc.calculate_values(tumor_mask, spacing)
                # Probable conversion from cm³ to mm³ (1 cm³ = 1000 mm³)
                volume_pred_A *= 1000
            except Exception as e:
                print(f"Error in rc.calculate_values for {patient_id}-t{tumor_id}: {str(e)}")
                recist_pred_A, volume_pred_A, slice_idx_A = 0.0, 0.0, 0
            
            # Method B: Alternative calculation
            try:
                volume_pred_B, diam1, diam2, diam3 = calculate_ellipsoid_volume(tumor_mask, spacing)
                recist_pred_B, slice_idx_B = calculate_max_diameter(tumor_mask, spacing)
            except Exception as e:
                print(f"Error in method B for {patient_id}-t{tumor_id}: {str(e)}")
                volume_pred_B, diam1, diam2, diam3, recist_pred_B, slice_idx_B = 0.0, 0.0, 0.0, 0.0, 0.0, 0
            
            # Update maximum RECIST values
            if recist_pred_A > max_recist_A: 
                max_recist_A = recist_pred_A
            if recist_pred_B > max_recist_B: 
                max_recist_B = recist_pred_B
            
            tumor_data.append({
                'patient_id': patient_id,
                'tumor_id': tumor_id,
                # Method A (rc.calculate_values)
                'recist_mm_A': recist_pred_A,
                'volume_mm3_A': volume_pred_A,
                'representative_slice_A': slice_idx_A,
                # Method B (alternative)
                'recist_mm_B': recist_pred_B,
                'volume_mm3_B': volume_pred_B,
                'diameter1_mm_B': diam1,
                'diameter2_mm_B': diam2,
                'diameter3_mm_B': diam3,
                'representative_slice_B': slice_idx_B,
                # Additional information
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
        print(f"Global error for {patient_id}: {str(e)}")
        return None, None

# Main pipeline
if __name__ == "__main__":
    start_time = time.time()
    
    # Patient list
    patients = [
        os.path.join(BASE_PATH, d) 
        for d in os.listdir(BASE_PATH) 
        if os.path.isdir(os.path.join(BASE_PATH, d))
    ]
    print(f"Processing {len(patients)} patients...")
    
    # Parallel processing
    n_jobs = max(1, os.cpu_count() - 2)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_patient)(patient_dir) 
        for patient_dir in patients
    )
    
    # Result aggregation
    all_tumors = []
    all_summaries = []
    
    for tumor_data, summary in results:
        if tumor_data and summary:
            all_tumors.extend(tumor_data)
            all_summaries.append(summary)
    
    # Save results
    if all_tumors:
        tumor_df = pd.DataFrame(all_tumors)
        summary_df = pd.DataFrame(all_summaries)
        
        # Output paths
        CSV_TUMOR_PATH = 'Model_1/T_Stage_Classification/tumor_volume_analysis_comparison.csv'
        CSV_SUMMARY_PATH = 'Model_1/T_Stage_Classification/patient_volume_summaries_comparison.csv'
        
        # CSV save
        tumor_df.to_csv(CSV_TUMOR_PATH, index=False)
        summary_df.to_csv(CSV_SUMMARY_PATH, index=False)
        
        # Final report
        duration = time.time() - start_time
        report = f"""
        COMPARATIVE VOLUMETRIC ANALYSIS REPORT
        ======================================
        Patients processed: {len(summary_df)}
        Tumors analyzed: {len(tumor_df)}
        Total time: {duration:.1f} seconds
        Average time per patient: {duration/max(1, len(summary_df)):.2f} sec
        
        KEY STATISTICS (METHOD A):
        - Average maximum RECIST: {summary_df['max_recist_mm_A'].mean():.1f} mm
        - Average tumor volume: {tumor_df['volume_mm3_A'].mean():.1f} mm³
        - Median volume: {tumor_df['volume_mm3_A'].median():.1f} mm³
        
        KEY STATISTICS (METHOD B):
        - Average maximum RECIST: {summary_df['max_recist_mm_B'].mean():.1f} mm
        - Average tumor volume: {tumor_df['volume_mm3_B'].mean():.1f} mm³
        - Median volume: {tumor_df['volume_mm3_B'].median():.1f} mm³
        
        RECIST/VOLUME CORRELATION (METHOD A):
        - Correlation: {tumor_df['recist_mm_A'].corr(tumor_df['volume_mm3_A']):.3f}
        
        RECIST/VOLUME CORRELATION (METHOD B):
        - Correlation: {tumor_df['recist_mm_B'].corr(tumor_df['volume_mm3_B']):.3f}
        
        INTER-METHOD CORRELATION:
        - RECIST: {tumor_df['recist_mm_A'].corr(tumor_df['recist_mm_B']):.3f}
        - Volume: {tumor_df['volume_mm3_A'].corr(tumor_df['volume_mm3_B']):.3f}
        
        VOLUME DISTRIBUTION (METHOD A):
        - Min: {tumor_df['volume_mm3_A'].min():.1f} mm³
        - Max: {tumor_df['volume_mm3_A'].max():.1f} mm³
        - Standard deviation: {tumor_df['volume_mm3_A'].std():.1f} mm³
        
        VOLUME DISTRIBUTION (METHOD B):
        - Min: {tumor_df['volume_mm3_B'].min():.1f} mm³
        - Max: {tumor_df['volume_mm3_B'].max():.1f} mm³
        - Standard deviation: {tumor_df['volume_mm3_B'].std():.1f} mm³
        
        NODULE DISTRIBUTION:
        {summary_df['num_nodules'].value_counts().sort_index()}
        """
        print(report)
        
        with open('volume_comparison_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Results saved in:")
        print(f"- Tumor details: {CSV_TUMOR_PATH}")
        print(f"- Patient summaries: {CSV_SUMMARY_PATH}")
        
        # Additional analysis
        print("\nRECIST vs VOLUME ANALYSIS (Method A):")
        print(tumor_df[['recist_mm_A', 'volume_mm3_A']].describe())
        
        print("\nRECIST vs VOLUME ANALYSIS (Method B):")
        print(tumor_df[['recist_mm_B', 'volume_mm3_B']].describe())
        
        # Option: Result visualization

        import matplotlib.pyplot as plt
        
        # RECIST comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.scatter(tumor_df['recist_mm_A'], tumor_df['recist_mm_B'], alpha=0.5)
        max_recist = max(tumor_df['recist_mm_A'].max(), tumor_df['recist_mm_B'].max())
        plt.plot([0, max_recist], [0, max_recist], 'r--')
        plt.title('RECIST Measurement Comparison')
        plt.xlabel('RECIST Method A (mm)')
        plt.ylabel('RECIST Method B (mm)')
        plt.grid(True)
        
        # Volume comparison
        plt.subplot(122)
        plt.scatter(tumor_df['volume_mm3_A'], tumor_df['volume_mm3_B'], alpha=0.5)
        max_vol = max(tumor_df['volume_mm3_A'].max(), tumor_df['volume_mm3_B'].max())
        plt.plot([0, max_vol], [0, max_vol], 'r--')
        plt.title('Tumor Volume Comparison')
        plt.xlabel('Volume Method A (mm³)')
        plt.ylabel('Volume Method B (mm³)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png')
        plt.show()
        
        print("Graph saved: method_comparison.png")
        
    else:
        print("No valid data to export")