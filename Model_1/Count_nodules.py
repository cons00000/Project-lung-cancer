import numpy as np
from tqdm import tqdm
from skimage.measure import label
import sys
sys.path.append('/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Segmentation')
sys.path.append('/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/T_Stage_Classification')
from TheDuneAI import ContourPilot as cp
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

# Imports for Xplique
import requests
from PIL import Image
import xplique
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad, SquareGrad,
                                  Occlusion, Rise, SobolAttributionMethod, HsicAttributionMethod)
from xplique.plots import plot_attributions
from xplique.metrics import Deletion, MuFidelity, Insertion, AverageStability
from xplique.plots.metrics import barplot

import shap

BATCH_SIZE = 8

import SimpleITK as sitk
import os
import pandas as pd
import re
import openpyxl

def count_connected_volumes(binary_mask, min_voxels=50, connectivity=3):
    """
    Counts connected volumes in a 3D binary mask with size filtering.
    
    Args:
        binary_mask (np.ndarray): 3D binary mask (slices, height, width)
        min_voxels (int): Minimum voxels required to count as a volume
        connectivity (int): 3D connectivity (1=6-connectivity, 2=18, 3=26)
    
    Returns:
        int: Number of connected volumes meeting size threshold
    """
    labeled_volume = label(binary_mask, connectivity=connectivity)
    volumes = []
    for label_val in np.unique(labeled_volume):
        if label_val == 0:  # Skip background
            continue
        volume_size = np.sum(labeled_volume == label_val)
        if volume_size >= min_voxels:
            volumes.append(label_val)
    return len(volumes)

def process_volume_slices(model, volume_3d, threshold=0.5, class_index=1):
    """
    Processes a 3D volume slice-by-slice and reconstructs 3D mask
    
    Args:
        model: Segmentation model expecting 3D input (H, W, 1)
        volume_3d (np.ndarray): Input volume (slices, height, width)
        threshold (float): Confidence threshold for binarization
        class_index (int): Class index for multi-class segmentation
    
    Returns:
        tuple: (binary_mask_3d, num_volumes)
    """
    # Initialize empty mask array
    mask_3d = np.zeros(volume_3d.shape, dtype=np.uint8)
    
    # Process each slice individually
    for i in range(volume_3d.shape[0]):
        # Get current slice and add channel dimension
        slice_img = volume_3d[i, :, :]
        input_slice = slice_img[np.newaxis, ..., np.newaxis].astype(np.float32) / 255.0
        
        # Predict segmentation
        pred = model.predict(input_slice, verbose=0)
        
        # Handle different output types
        if pred.shape[-1] == 1:  # Binary segmentation
            slice_mask = (pred[0, ..., 0] > threshold).astype(np.uint8)
        else:  # Multi-class segmentation
            slice_mask = (np.argmax(pred, axis=-1)[0] == class_index).astype(np.uint8)
        
        # Store in 3D mask
        mask_3d[i, :, :] = slice_mask
    
    # Count connected volumes in the reconstructed 3D mask
    num_volumes = count_connected_volumes(mask_3d)
    
    return mask_3d, num_volumes

# Main processing pipeline
def process_patients(model, patient_gen, min_voxels=50, threshold=0.5):
    """
    Process all patients from generator and return volume counts
    
    Args:
        model: Loaded segmentation model
        patient_gen: Generator yielding (img, metadata, filename, params)
        min_voxels: Minimum voxels for volume counting
        threshold: Confidence threshold for binarization
    
    Returns:
        dict: {filename: (num_volumes, mask_shape)}
    """
    results = {}
    gen_with_progress = tqdm(patient_gen, desc='Processing Patients')
    
    for img, _, filename, params in gen_with_progress:
        # Remove batch and channel dimensions
        vol_3d = img.squeeze()
        
        # Process entire volume slice-by-slice
        mask_3d, num_volumes = process_volume_slices(
            model=model,
            volume_3d=vol_3d,
            threshold=threshold
        )
        
        # Store results
        results[filename] = (num_volumes, mask_3d.shape)
        
        # Update progress bar
        gen_with_progress.set_postfix(
            file=filename[:15], 
            volumes=num_volumes,
            shape=vol_3d.shape
        )
    
    return results

# Usage example
if __name__ == "__main__":
    # Initialize model
    model_path = '/usr/users/pred_lung_cancer/piquet_con/Project-lung-cancer/Model_1/Segmentation/model_files'
    path_to_test_data = '/mounts/Datasets4/pred_lung_cancer/NIH dataset_raw/NRRD/converted_nrrds/'
    save_path = '/mounts/Datasets4/pred_lung_cancer/NIH dataset_raw/Processed'

    model = cp(model_path, path_to_test_data, save_path, verbosity=True)
    
    # Process patients
    results = process_patients(
        model=model.model1,
        patient_gen=model.Patients_gen,
        min_voxels=50,
        threshold=0.5  # Adjust confidence threshold as needed
    )
    
    # Print summary
    print("\nVolume Count Summary:")
    for file, (count, shape) in results.items():
        print(f"- {file}: {count} volumes (Mask shape: {shape})")
    
    # Optional: Save results to CSV
    df = pd.DataFrame.from_dict(results, orient='index', columns=['Volume_Count', 'Mask_Shape'])
    df.index.name = 'Filename'
    df.to_csv('volume_counts.csv')
    print("\nResults saved to volume_counts.csv")