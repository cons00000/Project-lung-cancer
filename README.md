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
- [Model implementation and analysis](#model-implementation-and-analysis)
  - [DuneAI for Segmentation](#duneai-for-segmentation)
  - [T-Stage Classification Logic](#t-stage-classification-logic)
  - [Explainable AI (XAI) Analysis](#explainable-ai-xai-analysis)
  - [Visualizing the Model’s Focus in 3D](#visualizing-the-model’s-focus-in-3d)
  - [Classification Performance](#classification-performance)
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
├── Data-analysis/
│   ├── data_analysis.ipynb          # Data analysis notebook
│   └── requirements.txt             # Dependencies for analysis
│
├── Figures/                         # Directory for graphs and visualizations
│
├── Model_1/                         # First processing model
│   ├── Data_preparation/            # Data preparation module
│   │   ├── AnalysisBox.py          
│   │   ├── data_preparation.ipynb   # Run the data preparation
│   │   ├── DataSet.py             
│   │   ├── FeaturesSet.py          
│   │   ├── GenerateResultBox.py    
│   │   ├── ResultSet.py            
│   │   └── ToolBox.py              
│   │
│   ├── requirements_data_prep.txt   # Environnement dependencies for data preparation 
│   ├── requirements_seg.slurm       # SLURM configuration for segmentation
│   │
│   ├── Segmentation/                # Segmentation module (DuneAI)
│   │   ├── Generator_v1.py         
│   │   ├── lung_extraction_funcs_13_09.py  
│   │   ├── model_files/            
│   │   ├── segmentation.py         # Main segmentation script
│   │   ├── segmentation.sh         # Run the segmentation
│   │   └── TheDuneAI.py           
│   │
│   ├── T_Stage_Classification/      # T-stage classification
│   │   ├── Tumor_size_measures.py   # Measurements of the tumor's size
│   │   ├── Tumor_size_measures.sh   # Run the alternative measurements script
│   │   ├── Stats.ipynb              # Notebook to run the stats for the measurements
│   │   ├── recist_and_volume_calculator.py  
│   │
│   └── Xai/                       # Explainable AI (XAI)
│       ├── Local_XAI.ipynb        # Local XAI notebook
│       └── Local_XAI.sh           # Run local XAI script
│
├── Model_2 (weights lacking)/      # Second model (weights missing)
│   └── UnSegMedGAT/                # UnSegMedGAT architecture
│
└── README.md                        
```

## Getting Started

### Prerequisites
- Python 3.8+
- Conda (`Miniconda` or `Anaconda`)
- A GPU (necessary for the segmentation but not the data analysis)

### Installation and Data Setup

**Step 1: Clone the Repository**
```bash
git clone https://github.com/cons000000/project-lung-cancer.git
cd project-lung-cancer
```

**Step 2: Set Up Conda Environments**
Different environments are required due to dependency constraints.

#### Data-analysis :

```bash
# 1. Environment for Data Preparation
conda create --name analysis_env python=3.8 -y
conda activate analysis_env
pip install -r Data-analysis/requirements.txt
```

#### Model_1 :

```bash
# 1. Environment for Data Preparation
conda create --name dataprep python=3.8 -y
conda activate dataprep
pip install -r Model_1/requirements_data_prep.txt
```

```bash
# 2. Environment for Segmentation, T_Stage_Classification & XAI
sbatch Model_1/Segmentation/requirements_seg.slurm
```

**Step 3: Download the Dataset**
This project uses the **Lung-PET-CT-Dx** dataset from The Cancer Imaging Archive (TCIA).

1.  Navigate to the [TCIA Data Access Page](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) and accept the data usage policy.
2.  **Download Clinical Data:** Download the "Clinical Data" spreadsheet directly from the page.
3.  **Download Image Data:**
    *   Image data must be downloaded using the **NBIA Data Retriever**. [Install it from the TCIA Wiki](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-DownloadingtheNBIADataRetriever).
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

Here are the labels generally used in radiology to caracteristize each of these stages. There is no mention of M2 or M3 as one can see (M-Stage). Regarding the N-Stage, it requires not only to identify a tumor but also to look if it has eventually metastasized and where. Given the complexity of this task, the present project is focusing on finding the true T-Stage.

| T (Tumor)            | N (Nodes)               | M (Metastasis)          |
|----------------------|-------------------------|-------------------------|
| **is**: In situ     | **N0**: No nodes        | **M0**: No metastasis   |
| **T1** ≤3 cm         | **N1**: Ipsilateral hilar | **M1**: Distant        |
| ・T1a ≤1 cm          | **N2**: Mediastinal     | ・M1a: Contralateral lung |
| ・T1b >1-2 cm        | **N3**: Contralateral   | ・M1b: Single distant   |
| ・T1c >2-3 cm        |                         | ・M1c: Multiple distant |
| **T2** >3-5 cm       |                         |                         |
| ・T2a >3-4 cm        |                         |                         |
| ・T2b >4-5 cm        |                         |                         |
| **T3** >5-7 cm       |                         |                         |
| **T4** >7 cm         |                         |                         |


## Model implementation and analysis

### DuneAI for Segmentation
**DuneAI** is a deep learning model designed for the automated detection and segmentation of non-small cell lung cancer (NSCLC) in CT images. We use the **[precision-medicine-toolbox](https://github.com/primakov/precision-medicine-toolbox)** for data preprocessing.

- **Preprocessing:** DICOM images are converted to NRRD using the precision medicine toolbox.
- **Input Format:** NRRD files.

### T-Stage Classification Logic
The model first segments the tumor and outputs a mask. 

We then calculate the size of the mask to give the tumor size. Ultimatly it is mapped to a clinical T-stage according to established guidelines.

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

### Explainable AI (XAI) Analysis

Just below is the data for one patient and the segmentation's result after using DuneAI.

<p align="center">
  <img src="Figures/outputwomask.png" alt="Model segmentation" width="70%"/>
  <br>
  <b>Figure 2:</b> Original CT for one patient. The 3D image is a sequence of 36 slices.
</p>

<p align="center">
  <img src="Figures/output.png" alt="Model segmentation" width="70%"/>
  <br>
  <b>Figure 3:</b> Example of DuneAI segmentation results showing the original image overlayed with the predicted mask.
</p>

To truly trust a model, we must understand *how* it arrives at its conclusions. Since the model analyzes the CT scan slice by slice, we can apply XAI methods to each slice to create a sequence of attribution maps. This approach provides a pseudo-3D, volumetric understanding of the model's decision-making process, allowing us to see how its focus evolves across the depth of the tumor.

**Figure 4: Volumetric XAI Attribution Maps**
*The table below visualizes the model's focus across all slices of a patient's scan for different XAI methods.*

| Method | Visualization | Description |
| :--- | :--- | :--- |
| **Saliency Map** | <div style="display:flex; flex-direction:column"> <img src="Figures/saliency1.png" alt="Saliency explanation" style="width:100%"> <img src="Figures/saliency2.png" alt="Saliency explanation" style="width:100%"> <img src="Figures/saliency3.png" alt="Saliency explanation" style="width:100%"> <img src="Figures/saliency4.png" alt="Saliency explanation" style="width:100%"> <img src="Figures/saliency5.png" alt="Saliency explanation" style="width:100%"> <img src="Figures/saliency6.png" alt="Saliency explanation" style="width:100%"> </div> | **Raw pixel influence.** Shows the gradient of the output with respect to the input pixels. It's fast but can be noisy. |
| **Gradient × Input** | <div style="display:flex; flex-direction:column"> <img src="Figures/gradientinput1.png" alt="Gradient x Input explanation" style="width:100%"> <img src="Figures/gradientinput2.png" alt="Gradient x Input explanation" style="width:100%"> <img src="Figures/gradientinput3.png" alt="Gradient x Input explanation" style="width:100%"> <img src="Figures/gradientinput4.png" alt="Gradient x Input explanation" style="width:100%"> <img src="Figures/gradientinput5.png" alt="Gradient x Input explanation" style="width:100%"> <img src="Figures/gradientinput6.png" alt="Gradient x Input explanation" style="width:100%"> </div> | **Influence combined with pixel intensity.** Weights the gradients by the input pixel values, highlighting influential bright/dark areas. |
| **Integrated Gradients** | <div style="display:flex; flex-direction:column"> <img src="Figures/integratedgradient1.png" alt="Integrated Gradients explanation" style="width:100%"> <img src="Figures/integratedgradient2.png" alt="Integrated Gradients explanation" style="width:100%"> <img src="Figures/integratedgradient3.png" alt="Integrated Gradients explanation" style="width:100%"> <img src="Figures/integratedgradient4.png" alt="Integrated Gradients explanation" style="width:100%"> <img src="Figures/integratedgradient5.png" alt="Integrated Gradients explanation" style="width:100%"> <img src="Figures/integratedgradient6.png" alt="Integrated Gradients explanation" style="width:100%"> </div> | **Stable, cumulative pixel importance.** Aggregates gradients along a path from a baseline (black) image to the input, providing more robust and less noisy attributions. |
| **SmoothGrad** | <div style="display:flex; flex-direction:column"> <img src="Figures/smoothgrad1.png" alt="SmoothGrad explanation" style="width:100%"> <img src="Figures/smoothgrad2.png" alt="SmoothGrad explanation" style="width:100%"> <img src="Figures/smoothgrad3.png" alt="SmoothGrad explanation" style="width:100%"> <img src="Figures/smoothgrad4.png" alt="SmoothGrad explanation" style="width:100%"> <img src="Figures/smoothgrad5.png" alt="SmoothGrad explanation" style="width:100%"> <img src="Figures/smoothgrad6.png" alt="SmoothGrad explanation" style="width:100%"> </div> | **Noise-reduced explanation.** Averages saliency maps over multiple noisy copies of the input to produce a cleaner, more stable visualization. |
| **SquareGrad** | <div style="display:flex; flex-direction:column"> <img src="Figures/squaregrad1.png" alt="SquareGrad explanation" style="width:100%"> <img src="Figures/squaregrad2.png" alt="SquareGrad explanation" style="width:100%"> <img src="Figures/squaregrad3.png" alt="SquareGrad explanation" style="width:100%"> <img src="Figures/squaregrad4.png" alt="SquareGrad explanation" style="width:100%"> <img src="Figures/squaregrad5.png" alt="SquareGrad explanation" style="width:100%"> <img src="Figures/squaregrad6.png" alt="SquareGrad explanation" style="width:100%"> </div> | **Magnitude of influence.** Similar to SmoothGrad, but focuses on the magnitude of gradients, highlighting impactful regions regardless of positive/negative influence. |
| **VarGrad** | <div style="display:flex; flex-direction:column"> <img src="Figures/vargrad1.png" alt="VarGrad explanation" style="width:100%"> <img src="Figures/vargrad2.png" alt="VarGrad explanation" style="width:100%"> <img src="Figures/vargrad3.png" alt="VarGrad explanation" style="width:100%"> <img src="Figures/vargrad4.png" alt="VarGrad explanation" style="width:100%"> <img src="Figures/vargrad5.png" alt="VarGrad explanation" style="width:100%"> <img src="Figures/vargrad6.png" alt="VarGrad explanation" style="width:100%"> </div> | **Stability/uncertainty of model focus.** Measures the variance of gradients. Bright areas indicate regions where the model's focus is unstable or uncertain. |
| **Sobol Attribution** | <div style="display:flex; flex-direction:column"> <img src="Figures/SobolAttributionMethod1.png" alt="Sobol Attribution explanation" style="width:100%"> <img src="Figures/SobolAttributionMethod2.png" alt="Sobol Attribution explanation" style="width:100%"> <img src="Figures/SobolAttributionMethod3.png" alt="Sobol Attribution explanation" style="width:100%"> <img src="Figures/SobolAttributionMethod4.png" alt="Sobol Attribution explanation" style="width:100%"> <img src="Figures/SobolAttributionMethod5.png" alt="Sobol Attribution explanation" style="width:100%"> <img src="Figures/SobolAttributionMethod6.png" alt="Sobol Attribution explanation" style="width:100%"> </div> | **Importance including feature interactions.** A sophisticated method that captures not just individual pixel importance but also the contribution of interactions between pixels. |

### Classification Performance

This study compares two different T-stage classification methods to evaluate their effectiveness in predicting cancer stages:

1. **RECIST Method:** The original implementation that calculates tumor diameter using the RECIST (Response Evaluation Criteria in Solid Tumors) guidelines, measuring the longest diameter of the tumor. For each slice of the mask, the minimum circle including all points in computed and the output gives the maximum diameter of these circles.

2. **Ellipsoid Method:** An alternative approach that models the tumor as an ellipsoid for each slice. We then take the maxium of these diameters. 

#### RECIST Method Performance

The RECIST method's T-stage classification performance was evaluated using a confusion matrix.

<p align="center">
  <img src="Figures/recist_confusion_matrix.png" alt="RECIST Confusion Matrix" width="60%"/>
</p>
<p align="center">
  <b>Figure 5a:</b> Confusion Matrix for RECIST method showing model-predicted T-stages versus the medical identified T-stages.
</p>

**Analysis of RECIST Classification Performance:**

- **Strongest Performance:** The RECIST method demonstrates accuracy in identifying **T1c** stage tumors, correctly classifying 24 cases with high confidence, making it the most reliable staging category.

- **Boundary Sensitivity Challenges:** Adjacent stage confusion represents a significant limitation, with actual **T1c** cases frequently misclassified as **T1b** (13 cases) or **T2a** (20 cases). This pattern indicates the model correctly identifies tumor size ranges but struggles with precise diameter-based boundary determinations.

- **Systematic Stage Overestimation:** The method shows consistent upward staging bias, particularly problematic when reconciling general labels (T2, T3) with specific sub-classifications. Actual **T2** cases predominantly misclassify as **T2b** (10 cases), **T3** (16 cases), or **T4** (6 cases), suggesting conservative treatment implications.

- **Small Tumor Detection Failure:** Complete inability to correctly identify **T1a** cases represents a critical clinical limitation, consistently overestimating these smallest tumors as **T1b** or **T1c**, potentially leading to overtreatment of early-stage disease.

#### Ellipsoid Method Performance

The ellipsoid method's performance was similarly evaluated to provide a direct comparison with the RECIST approach.

<p align="center">
  <img src="Figures/ellipsoid_confusion_matrix.png" alt="Ellipsoid Confusion Matrix" width="60%"/>
</p>
<p align="center">
  <b>Figure 5b:</b> Confusion Matrix for Ellipsoid method showing model-predicted T-stages versus the actual T-stages from the dataset.
</p>

**Analysis of Ellipsoid Classification Performance:**

- **Superior Volumetric Precision:** The ellipsoid method leverages three-dimensional tumor volume assessment, providing more comprehensive staging information than single-diameter RECIST measurements, particularly beneficial for irregularly shaped tumors.

- **Balanced Stage Distribution:** Demonstrates strong diagonal performance across multiple stages with **T1c** (23 correct), **T2** (12 correct), and **T3** (13 correct) classifications, indicating more consistent accuracy across the staging spectrum.

- **Reduced Boundary Confusion:** While some adjacent-stage misclassification persists (14 T1b cases predicted as T1c), the overall pattern shows less systematic bias compared to RECIST, with more accurate intermediate stage differentiation.

- **Advanced Stage Limitations:** Limited **T4** representation (5 total cases) constrains performance assessment for advanced disease, though the method shows better preservation of staging granularity without the systematic overestimation seen in RECIST.

- **Clinical Advantage:** The volumetric approach provides superior staging accuracy in the clinically critical **T1c-T3** range where treatment decisions heavily depend on precise tumor burden assessment, offering more reliable guidance for therapeutic planning.

#### Comparative Method Analysis

A comprehensive comparison between both methods to reveal the difference between their ouputs and their relative performance.
<p align="center">
  <img src="Figures/method_comparison.png" alt="Outputs' correlations" width="90%"/>
</p>
<p align="center">
  <b>Figure 6:</b> Correlation between RECIST and Ellipsoid methods regarding the tumor's size and volume.
</p>

<p align="center">
  <img src="Figures/method_comparison_plots.png" alt="Method Comparison" width="90%"/>
</p>
<p align="center">
  <b>Figure 7:</b> Comprehensive comparison between RECIST and Ellipsoid methods showing (left) accuracy comparison, (middle) mean stage error comparison, and (right) error distribution by T-stage.
</p>

**Key Findings from Method Comparison:**

- **Overall Accuracy:** The Ellipsoid method achieves 34.90% accuracy compared to RECIST's 32.29%, representing a 2.60% improvement in classification performance.

- **Error Patterns:** The Ellipsoid method shows consistently lower mean stage errors across most T-stages, with the most significant improvements in T1b (0.166 reduction) and T2b (0.250 reduction) classifications.

- **Stage-Specific Performance:** Both methods perform best on T3 and T4 stages (lowest mean errors of 0.52-0.60 and 0.40 respectively), while struggling most with T1a stages (2.80 mean error). The "in situ" case is problematic since only tumors of a certain size can be identified by the model.

- **Clinical Implications:** The Ellipsoid method's volume-based approach provides more accurate tumor size assessment, particularly beneficial for borderline cases between T-stages.

**Performance Summary:**

| Metric | RECIST Method | Ellipsoid Method | Better Method |
|--------|---------------|------------------|---------------|
| Overall Accuracy | 32.29% | 34.90% | **Ellipsoid** |
| Mean Stage Error | 1.193 | 1.125 | **Ellipsoid** |
| Error Rate | 67.71% | 65.10% | **Ellipsoid** |
| T1b Mean Error | 1.083 | 0.917 | **Ellipsoid** |
| T1c Mean Error | 1.173 | 1.136 | **Ellipsoid** |
| T2b Mean Error | 1.125 | 0.875 | **Ellipsoid** |
| Standard Deviation | 1.197 | 1.200 | RECIST (slightly) |


## Limitations and Future Work

### Known Issues
- **Limited Model Comparison:** The analysis was restricted to **DuneAI**, as pre-trained weights for the alternative model (**UnSegMedGAT**) were not publicly available.

- **Dataset Imperfections:** A subset of image files in the public dataset were found to be corrupted or missing essential `z-spacing` metadata, requiring data cleaning and exclusion.

- **2D Slice-Based Analysis:** Only local 3D-like XAI could be applied to this model. No global XAI, model-agnostic method was found to explain it.

### Future Work  
- **Architecture Comparison:**  
  Acquire weights for `Model_2` (UnSegMedGAT) to conduct a comprehensive performance and explainability comparison against existing models.  

- **3D Explainability Methods:**  
  Implement advanced XAI techniques to generate volumetric explanations of model predictions beyond slice-level interpretability.  

- **Model Repository Expansion:**  
  Integrate additional state-of-the-art pre-trained segmentation models through literature review and community collaboration.  

- **Multi-volume Tumor Analysis:**  
  Develop a post-processing pipeline to:  
  1. Aggregate slice-wise predictions into 3D masks with relaxed size constraints  
  2. Enable segmentation of disconnected tumor volumes  
  3. Correlate volumetric findings with clinical N-Stage classifications  

- **Volumetric Assessment Metrics:**  
  Design a 3D evaluation metric incorporating tumor height/depth to overcome current 2D measurement limitations, enabling true spatial size analysis.  

## References

#### Dataset
> Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). *A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx) [Data set]*. The Cancer Imaging Archive. [https://doi.org/10.7937/TCIA.2020.NNC2-0461](https://doi.org/10.7937/TCIA.2020.NNC2-0461)

#### Models and Toolboxes
> Primakov, S. P., Ibrahim, A., van Timmeren, J. E., et al. (2022). Automated detection and segmentation of non-small cell lung cancer computed tomography images. *Nature Communications, 13*(1), 3423. [https://doi.org/10.1038/s41467-022-30841-3](https://doi.org/10.1038/s41467-022-30841-3)

> Primakov, S., Lavrova, E., Salahuddin, Z., Woodruff, H. C., & Lambin, P. (2022). *Precision-medicine-toolbox: An open-source python package for facilitation of quantitative medical imaging and radiomics analysis*. arXiv preprint arXiv:2202.13965. [https://arxiv.org/abs/2202.13965](https://arxiv.org/abs/2202.13965)

> Fel, T., Hervier, L., Vigouroux, D., et al. (2022). *Xplique: A Deep Learning Explainability Toolbox*. Workshop on Explainable Artificial Intelligence for Computer Vision (CVPR). [https://arxiv.org/abs/2206.04394](https://arxiv.org/abs/2206.04394)

> Mudit Adityaja, A., Shigwan, S. J., & Kumar, N. (2024). UnSegMedGAT: Unsupervised Medical Image Segmentation using Graph Attention Networks Clustering. arXiv preprint arXiv:2411.01966. [https://doi.org/10.48550/arXiv.2411.01966](https://doi.org/10.48550/arXiv.2411.01966)

#### Related Literature
> Ma, J., He, Y., Li, F., et al. (2024). Segment anything in medical images. *Nature Communications, 15*(1), 654. [https://doi.org/10.1038/s41467-024-44824-z](https://doi.org/10.1038/s41467-024-44824-z)