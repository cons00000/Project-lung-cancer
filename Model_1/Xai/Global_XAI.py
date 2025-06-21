import os
import re
import sys
import numpy as np
import tensorflow as tf
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
BACKGROUND_SAMPLES = 50  # More samples for better SHAP estimation
# ===================================

# Create output directory
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
print(f"SHAP results will be saved to: {SHAP_OUTPUT_DIR}")
print(f"Processing {NUM_PATIENTS_TO_PROCESS} patients")

# Initialize model
print("Initializing segmentation model...")
model = cp(MODEL_PATH, PATH_TO_TEST_DATA, SAVE_PATH, verbosity=True)

# Create a Keras model wrapper for SHAP
def create_slice_model(segmentation_model):
    """Create a Keras model that outputs tumor percentage for a single slice"""
    
    # Get the input shape from the segmentation model
    input_shape = segmentation_model.input_shape
    print(f"Segmentation model input shape: {input_shape}")
    print(f"Segmentation model output shape: {segmentation_model.output_shape}")
    
    # Create input layer
    inputs = tf.keras.Input(shape=input_shape[1:])  # Remove batch dimension
    
    # Get segmentation output
    seg_output = segmentation_model(inputs)
    
    # Remove channel dimension if needed
    if len(seg_output.shape) > 3 and seg_output.shape[-1] == 1:
        seg_output = tf.squeeze(seg_output, axis=-1)
        print(f"Shape after squeeze: {seg_output.shape}")
    
    # Calculate tumor percentage
    tumor_percentage = tf.reduce_mean(seg_output, axis=[1, 2])
    print(f"Tumor percentage output shape: {tumor_percentage.shape}")
    
    # Create and return the composed model
    slice_model = tf.keras.Model(inputs=inputs, outputs=tumor_percentage, name="slice_tumor_model")
    
    return slice_model

def save_slice_results(test_slice, shap_slice, filename, slice_idx):
    """Save visualization results for a single slice"""
    clean_filename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)
    slice_dir = os.path.join(SHAP_OUTPUT_DIR, f"slice_{slice_idx}")
    os.makedirs(slice_dir, exist_ok=True)
    
    # Determine color scale
    abs_max = np.max(np.abs(shap_slice)) or 1e-6
    
    # Save visualizations
    try:
        # Original slice
        plt.imsave(f"{slice_dir}/{clean_filename}_original.png", test_slice, cmap='gray')
        
        # SHAP heatmap
        plt.imsave(f"{slice_dir}/{clean_filename}_shap.png", shap_slice, 
                   cmap='coolwarm', vmin=-abs_max, vmax=abs_max)
        
        # Overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(test_slice, cmap='gray')
        plt.imshow(shap_slice, cmap='coolwarm', alpha=0.4)
        plt.axis('off')
        plt.savefig(f"{slice_dir}/{clean_filename}_overlay.png", 
                    bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        return True
    except Exception as e:
        print(f"Error saving slice {slice_idx}: {str(e)}")
        return False

# Prepare background data (2D slices)
print("Collecting background data for 2D slices...")
background_gen = iter(model.Patients_gen)
background_slices = []

while len(background_slices) < BACKGROUND_SAMPLES:
    try:
        data = next(background_gen)
        img = data[0]
        
        # Handle different dimensionalities
        if img.ndim == 3:  # (slices, H, W)
            for slice_idx in range(img.shape[0]):
                slice_data = img[slice_idx]
                if slice_data.ndim == 2:
                    slice_data = np.expand_dims(slice_data, axis=-1)
                background_slices.append(slice_data)
                if len(background_slices) >= BACKGROUND_SAMPLES:
                    break
                    
        elif img.ndim == 4:  # (batch, slices, H, W)
            for slice_idx in range(img.shape[1]):
                slice_data = img[0, slice_idx]
                if slice_data.ndim == 2:
                    slice_data = np.expand_dims(slice_data, axis=-1)
                background_slices.append(slice_data)
                if len(background_slices) >= BACKGROUND_SAMPLES:
                    break
                    
        elif img.ndim == 5:  # (batch, slices, H, W, C)
            for slice_idx in range(img.shape[1]):
                slice_data = img[0, slice_idx, :, :, 0]  # Remove channel dimension
                if slice_data.ndim == 2:
                    slice_data = np.expand_dims(slice_data, axis=-1)
                background_slices.append(slice_data)
                if len(background_slices) >= BACKGROUND_SAMPLES:
                    break
                    
    except StopIteration:
        # Restart generator if exhausted
        background_gen = iter(model.Patients_gen)
    except Exception as e:
        print(f"Background error: {str(e)}")

background_slices = np.array(background_slices[:BACKGROUND_SAMPLES])
print(f"Background slices shape: {background_slices.shape}")

# Create the slice model wrapper
print("Creating SHAP-compatible slice model...")
slice_model = create_slice_model(model.model1)
print("Created SHAP-compatible slice model")

# Create SHAP explainer (once, reusable for all slices)
print("Initializing SHAP explainer...")
explainer = shap.GradientExplainer(
    model=slice_model,
    data=background_slices,
    batch_size=1
)
print("Initialized SHAP explainer")

# Process patients
print("\nProcessing patients:")
processed_count = 0
patient_gen = iter(model.Patients_gen)

for idx in range(NUM_PATIENTS_TO_PROCESS):
    try:
        print(f"\n=== Processing patient {idx+1}/{NUM_PATIENTS_TO_PROCESS} ===")
        data = next(patient_gen)
        img, _, filename, _ = data
        filename = filename[0] if isinstance(filename, list) else filename
        
        # Print original image shape for debugging
        print(f"Original image shape: {img.shape}")
        
        # Handle different dimensionalities
        if img.ndim == 3:  # (slices, H, W)
            volume = img
        elif img.ndim == 4:  # (batch, slices, H, W)
            volume = img[0]
        elif img.ndim == 5:  # (batch, slices, H, W, C)
            volume = img[0, :, :, :, 0]
        else:
            print(f"Unsupported image dimensions: {img.shape}")
            continue
        
        # Ensure we have enough slices
        total_slices = volume.shape[0]
        if total_slices < NUM_SLICES_PER_PATIENT:
            print(f"Only {total_slices} slices available (needs {NUM_SLICES_PER_PATIENT})")
            continue
        
        # Select slices to process - ensure integers
        slice_indices = np.linspace(0, total_slices-1, NUM_SLICES_PER_PATIENT, dtype=int)
        print(f"Processing slices: {slice_indices}")
        
        # Process each slice
        for slice_idx in slice_indices:
            try:
                # Extract the specific slice
                slice_data = volume[slice_idx]
                
                # Add batch and channel dimensions
                if slice_data.ndim == 2:
                    slice_data = np.expand_dims(slice_data, axis=-1)
                slice_data = np.expand_dims(slice_data, axis=0)
                
                print(f"Slice data shape for SHAP: {slice_data.shape}")
                
                # Compute SHAP values for this slice
                shap_values = explainer.shap_values(slice_data)
                
                # Extract SHAP values for this slice
                shap_slice = shap_values[0, :, :, 0]  # Remove batch and channel dimensions
                
                # Predict tumor percentage for display
                mask = model.model1.predict(slice_data, batch_size=1, verbose=0)
                if mask.shape[-1] == 1:
                    mask = np.squeeze(mask, axis=-1)
                tumor_percentage = np.mean(mask)
                print(f"Slice {slice_idx}: tumor percentage = {tumor_percentage:.4f}")
                
                # Save results
                if save_slice_results(
                    test_slice=slice_data[0, :, :, 0],
                    shap_slice=shap_slice,
                    filename=filename,
                    slice_idx=slice_idx
                ):
                    print(f"Saved results for slice {slice_idx}")
                
            except Exception as e:
                print(f"Error processing slice {slice_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        processed_count += 1
        print(f"Finished processing patient {idx+1}")
        
    except StopIteration:
        print("No more patients available")
        break
    except Exception as e:
        print(f"Error processing patient: {str(e)}")
        import traceback
        traceback.print_exc()

print(f"\nProcessed {processed_count} of {NUM_PATIENTS_TO_PROCESS} patients")
print(f"Results saved to: {SHAP_OUTPUT_DIR}")