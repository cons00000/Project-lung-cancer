import os
import re
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt

# Add necessary paths
sys.path.append('/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Segmentation')
sys.path.append('/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/T_Stage_Classification')
from TheDuneAI import ContourPilot as cp

# ========== CONFIGURATION ==========
MODEL_PATH = '/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Segmentation/model_files'
PATH_TO_TEST_DATA = '/mounts/Datasets4/pred_lung_cancer/NIH dataset_raw/NRRD/converted_nrrds/'
SAVE_PATH = '/mounts/Datasets4/pred_lung_cancer/NIH dataset_raw/Processed'
SHAP_OUTPUT_DIR = '/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Xai/shap_results'

# Processing parameters
NUM_PATIENTS_TO_PROCESS = 10
NUM_SLICES_PER_PATIENT = 5
BACKGROUND_SAMPLES = 3
BATCH_SIZE = 8
USE_MODEL_CACHE = True  # Cache models to avoid rebuilding
# ===================================

# Create output directory
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
print(f"SHAP results will be saved to: {SHAP_OUTPUT_DIR}")
print(f"Processing {NUM_PATIENTS_TO_PROCESS} patients")

# Initialize model
print("Initializing segmentation model...")
model = cp(MODEL_PATH, PATH_TO_TEST_DATA, SAVE_PATH, verbosity=True)

# Model cache to avoid rebuilding
model_cache = {}

def build_tumor_percentage_model(segmentation_model, slice_indices):
    """Build model to calculate tumor percentages for specific slices"""
    # Create cache key
    cache_key = tuple(slice_indices)
    
    # Return cached model if available
    if USE_MODEL_CACHE and cache_key in model_cache:
        return model_cache[cache_key]
    
    # Build new model
    input_layer = tf.keras.Input(shape=(None, None, None, 1))
    outputs = []
    
    for idx in slice_indices:
        slice_i = tf.gather(input_layer, idx, axis=1)
        seg_output = segmentation_model(slice_i)
        
        # Handle output shapes
        if len(seg_output.shape) == 4 and seg_output.shape[-1] == 1:
            seg_output = tf.squeeze(seg_output, axis=-1)
        
        # Calculate tumor percentage
        tumor_percentage = tf.reduce_mean(seg_output, axis=[1, 2])
        outputs.append(tumor_percentage)
    
    # Combine outputs
    output_tensor = tf.stack(outputs, axis=1)
    new_model = tf.keras.Model(inputs=input_layer, outputs=output_tensor)
    
    # Add to cache
    if USE_MODEL_CACHE:
        model_cache[cache_key] = new_model
    
    return new_model

def prepare_background_data(patient_list, num_samples=3):
    """Prepare background data for SHAP"""
    background = []
    
    for i in range(min(num_samples, len(patient_list))):
        try:
            img = patient_list[i][0]  # Get image from tuple
            
            # Handle different dimensionalities
            if img.ndim == 4:  # (batch, slices, H, W)
                background.append(img[0] if img.shape[0] == 1 else img)
            elif img.ndim == 5:  # (batch, slices, H, W, C)
                background.append(img)
        except Exception as e:
            print(f"Background sample error: {str(e)}")
    
    return np.concatenate(background, axis=0) if background else None

def save_slice_results(test_slice, shap_slice, filename, slice_idx, output_dir):
    """Save visualization results for a single slice"""
    # Clean filename
    clean_filename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)
    slice_dir = os.path.join(output_dir, f"slice_{slice_idx}")
    os.makedirs(slice_dir, exist_ok=True)
    
    # Determine color scale
    abs_max = np.max(np.abs(shap_slice))
    vmin, vmax = (-abs_max, abs_max) if abs_max > 0 else (-1, 1)
    
    # Save visualizations
    try:
        # Original slice
        plt.imsave(os.path.join(slice_dir, f"{clean_filename}_original.png"), 
                   test_slice, cmap='gray')
        
        # SHAP heatmap
        plt.imsave(os.path.join(slice_dir, f"{clean_filename}_shap.png"),
                   shap_slice, cmap='coolwarm', vmin=vmin, vmax=vmax)
        
        # Overlay
        plt.figure(figsize=(10, 10))
        plt.imshow(test_slice, cmap='gray')
        plt.imshow(shap_slice, cmap='coolwarm', alpha=0.4)
        plt.axis('off')
        plt.savefig(os.path.join(slice_dir, f"{clean_filename}_overlay.png"),
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error saving slice {slice_idx}: {str(e)}")
        return False

def process_patient(data, background, patient_idx, total_patients):
    """Process a single patient"""
    if not data or len(data) < 4:
        print(f"Skipping patient {patient_idx}: invalid data format")
        return False
    
    img, _, filename, _ = data
    filename = filename[0] if isinstance(filename, list) else filename
    
    print(f"\nProcessing patient {patient_idx+1}/{total_patients}: {filename}")
    
    # Prepare test sample
    test_sample = img[np.newaxis, ...] if img.ndim == 4 else img
    if test_sample.shape[0] > 1:
        test_sample = test_sample[:1]  # Use first in batch
        
    total_slices = test_sample.shape[1]
    if total_slices < NUM_SLICES_PER_PATIENT:
        print(f"Skipping: only {total_slices} slices available (needs {NUM_SLICES_PER_PATIENT})")
        return False
    
    # Select slices to process
    slice_indices = np.linspace(0, total_slices-1, NUM_SLICES_PER_PATIENT, dtype=int)
    print(f"  Slices: {slice_indices} (of {total_slices})")
    
    # Build model
    percentage_model = build_tumor_percentage_model(model.model1, slice_indices)
    
    # Compute SHAP values
    try:
        explainer = shap.GradientExplainer(percentage_model, background, batch_size=BATCH_SIZE)
        shap_values = explainer.shap_values(test_sample)
    except Exception as e:
        print(f"SHAP computation failed: {str(e)}")
        return False
    
    # Process each slice
    success_count = 0
    for slice_idx in slice_indices:
        test_slice = test_sample[0, slice_idx, :, :, 0]
        
        # Extract SHAP values
        try:
            if isinstance(shap_values, list):
                # Multi-output format
                shap_slice = shap_values[0][0, slice_idx, :, :, 0]  # First output only
            else:
                # Single array format
                shap_slice = shap_values[0, slice_idx, :, :, 0]
        except Exception:
            try:
                shap_slice = shap_values[0, slice_idx, :, :]
            except:
                print(f"  Error extracting SHAP for slice {slice_idx}")
                continue
        
        # Save results
        if save_slice_results(test_slice, shap_slice, filename, slice_idx, SHAP_OUTPUT_DIR):
            success_count += 1
    
    print(f"  Saved {success_count}/{NUM_SLICES_PER_PATIENT} slices")
    return success_count > 0

# Main processing pipeline
def main():
    # Get all patients
    all_patients = list(model.Patients_gen)
    print(f"Total patients available: {len(all_patients)}")
    
    # Select patients to process
    patients_to_process = all_patients[:min(NUM_PATIENTS_TO_PROCESS, len(all_patients))]
    
    # Prepare background data
    print("Collecting background data...")
    background = prepare_background_data(patients_to_process, BACKGROUND_SAMPLES)
    if background is None:
        print("Error: Could not collect background data")
        return
    
    print(f"Background shape: {background.shape}")
    
    # Process patients
    processed_count = 0
    for idx, data in enumerate(tqdm(patients_to_process, 
                                   total=len(patients_to_process), 
                                   desc='Processing patients')):
        if process_patient(data, background, idx, len(patients_to_process)):
            processed_count += 1
        
        # Clear memory every 2 patients
        if (idx + 1) % 2 == 0:
            tf.keras.backend.clear_session()
            if USE_MODEL_CACHE:
                model_cache.clear()
            print("  Memory cleared")
    
    print(f"\nSuccessfully processed {processed_count}/{len(patients_to_process)} patients")
    print(f"Results saved to: {SHAP_OUTPUT_DIR}")

if __name__ == "__main__":
    main()