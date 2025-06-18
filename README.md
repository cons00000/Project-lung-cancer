# Explainable AI for Lung Cancer Stage Classification

This project provides a comprehensive pipeline for applying, evaluating, and interpreting a pre-trained deep learning model for lung cancer stage classification. By leveraging post-hoc Explainable AI (XAI) techniques, we demystify the model's decision-making process, ensuring its predictions are not only accurate but also transparent, reliable, and clinically relevant.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Pipeline](#project-pipeline)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation and Data Setup](#installation-and-data-setup)
- [Usage Guide](#usage-guide)
- [Data Exploration: The TNM Staging System](#data-exploration-the-tnm-staging-system)
- [Model Methodology](#model-methodology)
  - [DuneAI for Segmentation](#duneai-for-segmentation)
  - [T-Stage Classification Logic](#t-stage-classification-logic)
- [Results and Performance](#results-and-performance)
  - [Segmentation Results](#segmentation-results)
  - [Classification Performance](#classification-performance)
- [Explainable AI (XAI) Analysis](#explainable-ai-xai-analysis)
  - [Visualizing the Model’s Focus in 3D](#visualizing-the-model’s-focus-in-3d)
  - [Interpreting and Comparing XAI Methods](#interpreting-and-comparing-xai-methods)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)

## Project Overview

Artificial intelligence has shown remarkable success in medical imaging, but the "black box" nature of many deep learning models remains a significant barrier to their adoption in clinical practice. This project tackles that challenge head-on. We create and document an end-to-end framework to analyze a pre-trained model for classifying lung cancer stages from CT and PET/CT scans.

Our core objectives are:
- **Apply** a state-of-the-art model to segment and classify lung cancer.
- **Explain** the model's predictions using a diverse suite of XAI methods to understand its internal logic.
- **Evaluate** the model's behavior and performance against established clinical standards (TNM staging).
- **Establish** a reproducible pipeline for the analysis and interpretation of medical imaging AI.

## Key Features
- **End-to-End Inference Pipeline:** A complete workflow from data ingestion and preprocessing to segmentation, prediction, and explanation.
- **State-of-the-Art Model Integration:** Utilizes the **DuneAI** model for tumor segmentation. An implementation for a second model (**UnSegMedGAT**) is included for future extension.
- **Deep Explainability:** Leverages a wide array of XAI techniques from the `Xplique` library for a holistic view of the model's decision-making process.
- **Clinically Grounded Analysis:** The entire analysis is framed around the **TNM staging system** to assess the model's practical utility and alignment with medical standards.

## Project Pipeline

Our methodology is an inference and analysis pipeline designed to scrutinize a pre-trained model's behavior.

**`Data Acquisition`** → **`Preprocessing & Segmentation`** → **`Prediction (Inference)`** → **`Explainability Analysis`**

## Repository Structure
```
project-lung-cancer/
├── NIH dataset_raw/              # Raw TCIA dataset (images and clinical data)
├── Data-analysis/                # Jupyter notebooks for initial data exploration
├── Figures/                      # All plots and visualizations generated for the README
├── Model_1/                      # Primary model (DuneAI)
│   ├── Data_preparation/         # Scripts for converting DICOM to NRRD
│   ├── Segmentation/             # Scripts for running tumor segmentation
│   ├── T_Stage_Classification/   # Scripts for classifying T-stage from segmentations
│   └── Xai/                      # Scripts for generating XAI attribution maps
├── Model_2/                      # Secondary model (UnSegMedGAT, weights unavailable)
├── Visualize_lung_mask/          # Utility scripts for visualization
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.8+
- Conda (`Miniconda` or `Anaconda`)

### Installation and Data Setup

**Step 1: Clone the Repository**
```bash
git clone https://github.com/cons000000/project-lung-cancer.git
cd project-lung-cancer
```

**Step 2: Set Up Conda Environments**
Two separate environments are required due to dependency constraints.

```bash
# 1. Environment for Data Preparation
conda create --name lung-dataprep python=3.8 -y
conda activate lung-dataprep
pip install -r Model_1/Data_preparation/requirements.txt

# 2. Environment for Segmentation & XAI
conda create --name lung-segment python=3.10 -y
conda activate lung-segment
pip install -r Model_1/Segmentation/requirements.txt
```

**Step 3: Download the Dataset**
This project uses the **Lung-PET-CT-Dx** dataset from The Cancer Imaging Archive (TCIA).

1.  Navigate to the [TCIA Data Access Page](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) and accept the data usage policy.
2.  **Download Clinical Data:** Download the "Clinical Data" spreadsheet directly from the page.
3.  **Download Image Data:**
    *   Image data must be downloaded using the **NBIA Data Retriever**. [Install it from the TCIA Wiki](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Installation+and+Usage).
    *   On the TCIA page, add the image collection to your cart, and download the manifest file (`.tcia`).
    *   Open the NBIA Data Retriever, load the manifest file, and begin the download.
4.  **Organize Files:** Create a directory named `NIH dataset_raw/` at the project root. Place the downloaded clinical data file and the folder of patient images inside it.

## Usage Guide
Follow the notebooks in sequence, ensuring you activate the correct Conda environment for each stage.

1.  **Data Preparation (`conda activate lung-dataprep`)**
    - Run the notebooks in `Model_1/Data_preparation/` to process the raw DICOM files into the NRRD format required by the model.
2.  **Segmentation and Analysis (`conda activate lung-segment`)**
    - Run the notebook in `Model_1/Segmentation/` to perform tumor segmentation on the NRRD files using the DuneAI model.
    - Proceed to `Model_1/T_Stage_Classification/` to classify the T-stage based on the segmentation output.
    - Finally, run the notebooks in `Model_1/Xai/` to generate and visualize the explainability maps.

## Data Exploration: The TNM Staging System

Our analysis is grounded in the clinical **TNM Staging System**, the global standard for classifying the anatomical extent of cancer.

| Aspect         | TNM Staging                                 | Histopathological Grading          |
| :------------- | :------------------------------------------ | :--------------------------------- |
| **Focus**      | Anatomical spread of the tumor              | Cellular appearance & aggressiveness |
| **Components** | **T** (Tumor size), **N** (Nodes), **M** (Metastasis) | Differentiation (Grade G1-G3)      |

The **Lung-PET-CT-Dx** dataset provides labels for T, N, and M stages. The distribution of these labels is shown below.

<p align="center">
  <img src="Figures/chart2.svg" alt="T-Stage distribution" width="32%"/>
  <img src="Figures/chart1.svg" alt="N-Stage distribution" width="32%"/>
  <img src="Figures/chart3.svg" alt="M-Stage distribution" width="32%"/>
</p>
<p align="center">
  <b>Figure 1:</b> Distribution of ground truth labels for T-Stage, N-Stage, and M-Stage in the dataset.
</p>

## Model Methodology

### DuneAI for Segmentation
**DuneAI** is a deep learning model designed for the automated detection and segmentation of non-small cell lung cancer (NSCLC) in CT images. It relies on the **[precision-medicine-toolbox](https://github.com/primakov/precision-medicine-toolbox)** for data preprocessing.

- **Input Format:** NRRD files.
- **Preprocessing:** DICOM images are converted to NRRD using the provided scripts.

### T-Stage Classification Logic
The model first segments the tumor and calculates its size. This size is then mapped to a clinical T-stage according to established guidelines.

**Table 1: Mapping Tumor Size to T-Stage**
| T-Stage | Tumor Size                   |
| :------ | :--------------------------- |
| T1a     | ≤ 1 cm (≤ 10 mm)             |
| T1b     | > 1 cm and ≤ 2 cm (11–20 mm) |
| T1c     | > 2 cm and ≤ 3 cm (21–30 mm) |
| T2a     | > 3 cm and ≤ 4 cm (31–40 mm) |
| T2b     | > 4 cm and ≤ 5 cm (41–50 mm) |
| T3      | > 5 cm and ≤ 7 cm (51–70 mm) |
| T4      | > 7 cm (> 70 mm)             |

*Note: The ground truth dataset contains a mix of general labels (e.g., 'T2') and specific sub-stages (e.g., 'T2a'). This label granularity mismatch is a key challenge addressed in the performance analysis.*

## Results and Performance

### Segmentation Results
The DuneAI model processes each patient scan slice-by-slice to generate a 3D segmentation mask identifying the tumor.

<p align="center">
  <img src="Figures/output.png" alt="Model segmentation" width="70%"/>
  <br>
  <b>Figure 2:</b> Example of DuneAI segmentation results on a patient's middle CT slice, showing the original image, ground truth mask, and predicted mask.
</p>

### Classification Performance
The model's T-stage classification performance was evaluated using a confusion matrix. The matrix highlights the model's strengths and weaknesses in distinguishing between different cancer stages.

<p align="center">
  <img src="Figures/confusionmatrixoutput.png" alt="Confusion Matrix" width="60%"/>
</p>
<p align="center">
  <b>Figure 3:</b> Confusion Matrix showing model-predicted T-stages versus the actual T-stages from the dataset.
</p>

**Analysis of Classification Performance:**

- **Strongest Performance:** The model shows its highest confidence and accuracy in identifying the **T1c** stage, correctly classifying 24 cases.
- **Adjacent Stage Confusion:** A common pattern is the confusion between adjacent or near-adjacent stages, which is clinically understandable as it depends on precise size measurements. For example, actual **T1c** cases are frequently misclassified as **T1b** (13 cases) or **T2a** (20 cases). This indicates the model is identifying tumors of a similar size but the calculated diameter falls just across a classification boundary.
- **Handling of General vs. Specific Labels:** The ground truth contains general labels like '2' and '3', while the model predicts specific sub-stages. The model tends to overestimate the stage for these general labels; for example, actual **T2** cases are most often predicted as **T2b** (10), **T3** (16), or even **T4** (6). This highlights a challenge in reconciling different levels of label granularity.
- **Challenges with Smallest Tumors:** The model struggles significantly with the smallest tumors, failing to correctly identify any **T1a** cases, often classifying them as larger stages like **T1b** or **T1c**.

## Explainable AI (XAI) Analysis

### Visualizing the Model’s Focus in 3D

To truly trust a model, we must understand *how* it arrives at its conclusions. Since the model analyzes the CT scan slice by slice, we can apply XAI methods to each slice to create a sequence of attribution maps. This approach provides a pseudo-3D, volumetric understanding of the model's decision-making process, allowing us to see how its focus evolves across the depth of the tumor.

**Figure 4** presents these sequences for various XAI methods. The colored background highlights the slices containing the segmented tumor, where the model's attention is expected to be highest.

**Figure 4: Volumetric XAI Attribution Maps**
*The table below visualizes the model's focus across all relevant slices of a patient's scan for different XAI methods.*

| Method | Visualization | Description |
| :--- | :--- | :--- |
| **Saliency Map** | <img src="Figures/saliencyoutput.png" alt="Saliency explanation" width="100%"> | **Raw pixel influence.** Shows the gradient of the output with respect to the input pixels. It's fast but can be noisy. |
| **Gradient × Input** | <img src="Figures/gradientinputoutput.png" alt="Gradient x Input explanation" width="100%"> | **Influence combined with pixel intensity.** Weights the gradients by the input pixel values, highlighting influential bright/dark areas. |
| **Integrated Gradients** | <img src="Figures/integratedgradientsoutput.png" alt="Integrated Gradients explanation" width="100%"> | **Stable, cumulative pixel importance.** Aggregates gradients along a path from a baseline (black) image to the input, providing more robust and less noisy attributions. |
| **SmoothGrad** | <img src="Figures/smoothgradoutput.png" alt="SmoothGrad explanation" width="100%"> | **Noise-reduced explanation.** Averages saliency maps over multiple noisy copies of the input to produce a cleaner, more stable visualization. |
| **SquareGrad** | <img src="Figures/squaregradoutput.png" alt="SquareGrad explanation" width="100%"> | **Magnitude of influence.** Similar to SmoothGrad, but focuses on the magnitude of gradients, highlighting impactful regions regardless of positive/negative influence. |
| **VarGrad** | <img src="Figures/vargradoutput.png" alt="VarGrad explanation" width="100%"> | **Stability/uncertainty of model focus.** Measures the variance of gradients. Bright areas indicate regions where the model's focus is unstable or uncertain. |
| **Sobol Attribution** | <img src="Figures/SobolAttributionMethodoutput.png" alt="Sobol Attribution explanation" width="100%"> | **Importance including feature interactions.** A sophisticated method that captures not just individual pixel importance but also the contribution of interactions between pixels. |

### Interpreting and Comparing XAI Methods
By analyzing these explanation sequences, we can build a more robust understanding of the model's behavior than is possible from a single slice.

- **Confirming Consistent Focus:** The primary benefit of this volumetric view is the ability to check for consistency. Across all methods, we can observe that the model's attention (the colored highlights) is sharply concentrated on the tumor region and tracks it accurately from slice to slice. This provides strong evidence that the model has learned the correct anatomical features.

- **Assessing Stability Across Slices:** Methods like **Integrated Gradients** and **SmoothGrad** show a stable and clean focus that evolves smoothly with the tumor's shape across slices. This is a hallmark of a robust model.

- **Identifying Uncertainty:** **VarGrad** is particularly powerful in this context. A low variance (darker color) that is consistent across the tumor slices indicates the model is highly confident in its focus. If certain slices showed high variance (brighter colors), it could point to ambiguous features in those slices that the model finds difficult to interpret.

**Synthesis:** Viewing the explanations as a full sequence transforms the analysis. Instead of just confirming *what* the model is looking at on a single slice, we can assess the *stability and consistency* of its focus across the entire tumor volume. The strong and consistent focus on the tumor pathology, as seen across multiple XAI methods, significantly increases our trust in the model's underlying reasoning and its potential for clinical application.

## Limitations and Future Work

### Known Issues
- **Limited Model Comparison:** The analysis was restricted to **DuneAI**, as pre-trained weights for the alternative model (**UnSegMedGAT**) were not publicly available.
- **Dataset Imperfections:** A subset of image files in the public dataset were found to be corrupted or missing essential `z-spacing` metadata, requiring data cleaning and exclusion.
- **2D Slice-Based Analysis:** While our XAI analysis provides a pseudo-3D view, the model itself still operates on 2D slices, which may not fully capture complex 3D volumetric context.

### Future Work
- **Comparative Model Analysis:** Acquire weights for `Model_2` (UnSegMedGAT) to enable a direct performance and explainability comparison between different architectures.
- **True 3D Explainability:** Implement XAI methods designed specifically for 3D convolutional networks to generate a true 3D attribution map, moving beyond the slice-by-slice approximation.
- **Expand Model Repository:** Survey recent literature to identify and integrate other publicly available pre-trained models for a broader comparative study.

## References

#### Dataset
> Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). *A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx) [Data set]*. The Cancer Imaging Archive. [https://doi.org/10.7937/TCIA.2020.NNC2-0461](https://doi.org/10.7937/TCIA.2020.NNC2-0461)

#### Models and Toolboxes
> Primakov, S. P., Ibrahim, A., van Timmeren, J. E., et al. (2022). Automated detection and segmentation of non-small cell lung cancer computed tomography images. *Nature Communications, 13*(1), 3423. [https://doi.org/10.1038/s41467-022-30841-3](https://doi.org/10.1038/s41467-022-30841-3)

> Primakov, S., Lavrova, E., Salahuddin, Z., Woodruff, H. C., & Lambin, P. (2022). *Precision-medicine-toolbox: An open-source python package for facilitation of quantitative medical imaging and radiomics analysis*. arXiv preprint arXiv:2202.13965. [https://arxiv.org/abs/2202.13965](https://arxiv.org/abs/2202.13965)

> Fel, T., Hervier, L., Vigouroux, D., et al. (2022). *Xplique: A Deep Learning Explainability Toolbox*. Workshop on Explainable Artificial Intelligence for Computer Vision (CVPR). [https://arxiv.org/abs/2204.13132](https://arxiv.org/abs/2204.13132)

#### Related Literature
> Ma, J., He, Y., Li, F., et al. (2024). Segment anything in medical images. *Nature Communications, 15*(1), 654. [https://doi.org/10.1038/s41467-024-44824-z](https://doi.org/10.1038/s41467-024-44824-z)