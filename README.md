Of course. I have corrected your version, respecting the structure and content you've laid out while fixing formatting, improving clarity, and ensuring consistency.

Here are the key corrections made:
*   **Accuracy:** Changed "unsupervised machine learning model" to "pre-trained deep learning model," as DuneAI is a supervised model.
*   **Formatting:**
    *   Fixed the broken Markdown table for the "Visual Explanation Array," restoring the intended layout.
    *   Properly formatted the "Repository Structure" with a code block.
    *   Cleaned up the "Installation and Data Setup" steps into a clear, numbered list.
*   **Clarity & Consistency:**
    *   Streamlined the introduction to the XAI section to avoid repetition.
    *   Corrected and standardized all figure numbering throughout the document.

Here is the corrected and polished version:

---

# Explainable AI for Lung Cancer Stage Classification

<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.8+-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-in%20progress-yellow.svg">
</p>

This project applies and analyzes a pre-trained deep learning model for lung cancer stage classification, with an emphasis on transparency through Explainable AI (XAI) techniques. By implementing post-hoc explanation methods, we gain crucial insights into the model's decision-making process, evaluating its clinical relevance and reliability.

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Project Pipeline](#project-pipeline)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Data Exploration](#data-exploration)
- [Model Analysis and Explainability](#model-analysis-and-explainability)
- [Performance Evaluation](#performance-evaluation)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [License](#license)

## Introduction

While AI models can achieve high accuracy in medical imaging tasks, their "black box" nature can limit clinical adoption. This project addresses this by creating a pipeline to apply, evaluate, and interpret a pre-trained model for lung cancer classification, making its predictions transparent.

Our objectives are to:
- Apply a pre-trained model to classify lung cancer stages from CT and PET/CT scans.
- Utilize a suite of post-hoc XAI methods to understand the model's predictive logic.
- Establish a reproducible framework for analyzing pre-trained medical imaging models.
- Document best practices for applying XAI to evaluate model behavior in a clinical context.

## Key Features
- **End-to-End Inference Pipeline:** From data preprocessing and segmentation to prediction and explanation using a pre-trained model.
- **State-of-the-Art Model Application:** Implements the **DuneAI** model for segmentation. An implementation for a second model (**UnSegMedGAT**) is included, though its pre-trained weights were not available.
- **Deep Explainability:** Leverages a wide array of XAI techniques to provide a holistic view of the model's decision-making.
- **Clinical Focus:** Grounds the analysis in established medical standards like TNM staging to assess practical utility.

## Project Pipeline

Our methodology is an inference and analysis pipeline focused on a pre-trained model:

**`Data Acquisition`** → **`Preprocessing & Segmentation`** → **`Prediction (Inference)`** → **`Explainability Analysis`**

## Getting Started

### Prerequisites
- Python 3.8+
- Conda ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution))
- Git

### Installation and Data Setup

**Step 1: Clone the Repository**
```bash
git clone https://github.com/cons000000/project-lung-cancer.git
cd project-lung-cancer
```

**Step 2: Set Up Conda Environments**
```bash
# Environment for Data Preparation
conda create --name lung-dataprep python=3.8 -y
conda activate lung-dataprep
pip install -r Model_1/Data_preparation/requirements.txt

# Environment for Segmentation & XAI
conda create --name lung-segment python=3.10 -y
conda activate lung-segment
pip install -r Model_1/Segmentation/requirements.txt
```

**Step 3: Download the Dataset**
This project uses the **Lung-PET-CT-Dx** dataset from TCIA.

> Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). *A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx) [Data set]*. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2020.NNC2-0461

1.  **Go to the [TCIA Data Access Page](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)** and accept the data usage agreement.
2.  **Download Clinical Data:** Download the "Clinical Data" file directly.
3.  **Download Image Data:**
    *   The image data must be acquired using the **NBIA Data Retriever**. [Download and install it from the TCIA Wiki](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Installation+and+Usage) if you haven't already.
    *   On the TCIA website, add the image collection to your cart and download the **manifest file** (`.tcia`).
    *   Open the NBIA Data Retriever, import the manifest file, and start the download.
4.  **Organize Files:** Create a `NIH dataset_raw/` directory at the project root and place the downloaded clinical data file and the folder of patient images inside it.

## Repository Structure
```
project-lung-cancer/
├── NIH dataset_raw/              # Raw data goes here
├── Data-analysis/                # Initial data exploration
├── Model_1/                      # Primary model (DuneAI)
│   ├── Data_preparation/
│   ├── Segmentation/
│   ├── T_Stage_Classification/
│   └── Xai/
├── Model_2/                      # Second model (weights unavailable)
└── Visualize_lung_mask/
├── README.md
```

## Usage
Activate the correct Conda environment before running the Jupyter notebooks in each directory.
- **For data preparation:** `conda activate lung-dataprep`
- **For segmentation and XAI:** `conda activate lung-segment`

## Data Exploration: TNM Staging

Our analysis is grounded in the clinical **TNM Staging System**.

| Aspect                 | TNM Staging                               | Histopathological Grading                  |
| ---------------------- | ----------------------------------------- | ------------------------------------------ |
| **Focus**              | Anatomical spread of the tumor            | Cellular appearance & aggressiveness       |
| **Components**         | **T** (Tumor size), **N** (Nodes), **M** (Metastasis) | Differentiation (Grade G1-G3)              |

### Dataset Distributions
<p align="center">
  <img src="chart2.svg" alt="T-Stage distribution" width="30%"/>
  <img src="chart1.svg" alt="N-Stage distribution" width="30%"/>
  <img src="chart3.svg" alt="M-Stage distribution" width="30%"/>
</p>
<p align="center">
  <b>Figure 1:</b> T-Stage Distribution &nbsp;&nbsp;&nbsp;&nbsp; <b>Figure 2:</b> N-Stage Distribution &nbsp;&nbsp;&nbsp;&nbsp; <b>Figure 3:</b> M-Stage Distribution
</p>

## Model Analysis and Explainability

### Segmentation Results
We utilized **DuneAI**, a pre-trained deep learning model for automated segmentation of non-small cell lung cancer (NSCLC).

- **DuneAI Paper:** Primakov, S.P., et al. *Nat Commun* 13, 3423 (2022).
- **Toolbox Used:** [precision-medicine-toolbox](https://github.com/primakov/precision-medicine-toolbox)

<p align="center">
  <img src="output.png" alt="Model segmentation" width="70%"/>
  <br>
  <b>Figure 4:</b> DuneAI segmentation results on a sample patient's middle CT slice.
</p>

### Explainable AI (XAI) Analysis

To understand *how* the model produced this segmentation, we generated attribution maps using various XAI methods from the [Xplique](https://github.com/deel-ai/xplique) library. Since segmentation is performed slice-by-slice, these explanations show which pixels were most influential for a given slice's prediction.

#### Visual Explanation Array

| Method                   | Visualization                                                                                                                                                                                            |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Saliency Map**         | <p align="center"><img src="saliencyoutput.png" alt="Saliency explanation" width="85%"><br><b>Figure 5:</b> Raw pixel influence.</p>                                                                         |
| **Integrated Gradients** | <p align="center"><img src="integratedgradientsoutput.png" alt="Integrated Gradients explanation" width="85%"><br><b>Figure 6:</b> Stable, cumulative pixel importance.</p>                                  |
| **Gradient × Input**     | <p align="center"><img src="gradientinputoutput.png" alt="Gradient x Input explanation" width="85%"><br><b>Figure 7:</b> Influence combined with pixel intensity.</p>                                       |
| **SmoothGrad**           | <p align="center"><img src="smoothgradoutput.png" alt="SmoothGrad explanation" width="85%"><br><b>Figure 8:</b> Noise-reduced explanation.</p>                                                              |
| **Sobol Attribution**    | <p align="center"><img src="SobolAttributionMethodoutput.png" alt="Sobol Attribution explanation" width="85%"><br><b>Figure 9:</b> Importance including feature interactions.</p>                           |
| **SquareGrad**           | <p align="center"><img src="squaregradoutput.png" alt="SquareGrad explanation" width="85%"><br><b>Figure 10:</b> Magnitude of influence, regardless of direction.</p>                                      |
| **VarGrad**              | <p align="center"><img src="vargradoutput.png" alt="VarGrad explanation" width="85%"><br><b>Figure 11:</b> Stability/uncertainty of the model's focus.</p>                                                 |

#### Interpretation and Comparison of XAI Methods
No single XAI method tells the whole story. By comparing their outputs, we build a more robust and reliable understanding of the model's behavior.

-   **Baseline Sensitivity (Saliency, Gradient × Input):** These methods (Fig. 5, 7) provide a direct but often noisy look at which pixels the model is sensitive to. They are useful as a starting point but can be misleading due to instability.

-   **Noise Reduction and Stability (SmoothGrad, Integrated Gradients):** SmoothGrad (Fig. 8) cleans up the noise from the basic Saliency map by averaging over slightly perturbed inputs, revealing a clearer underlying pattern. Integrated Gradients (Fig. 6) offers a more theoretically sound approach to attribution, providing a stable and reliable map of feature importance. Comparing Fig. 5 and Fig. 8 clearly shows the benefit of noise reduction.

-   **Focusing on Magnitude (SquareGrad):** SquareGrad (Fig. 10) is useful for identifying the most impactful regions without getting distracted by whether their influence is positive or negative. It helps confirm the absolute importance of the tumor area.

-   **Advanced Insights (Sobol, VarGrad):** These methods provide deeper analysis.
    -   **Sobol Attribution (Fig. 9)** goes beyond individual pixels to show how *interactions* between different image regions contribute to the prediction.
    -   **VarGrad (Fig. 11)** is unique because it measures the model's *consistency*. High-variance (brighter) areas indicate regions where the model's focus is unstable or uncertain, which could signal ambiguity in the input.

**Comparative Synthesis:** By looking at all methods together, we can draw stronger conclusions. When the tumor region is consistently highlighted across Saliency, Integrated Gradients, and SmoothGrad, we gain confidence that the model is correctly focusing on the relevant pathology. If VarGrad shows low variance in that same area, it further strengthens our trust in the model's stability.

## Performance Evaluation
The model's classification performance was evaluated using standard metrics.

<p align="center">
  <img src="confusionmatrixoutput.png" alt="Confusion Matrix" width="45%"/>
  <img src="distributionerrorsoutput.png" alt="Error Distribution" width="45%"/>
  <br>
  <b>Figure 12:</b> Confusion Matrix (left) and Distribution of Prediction Errors (right).
</p>

## Limitations and Future Work

### Known Issues
- **Limited Model Comparison:** Only the DuneAI model was fully implementable due to the unavailability of public weights for other models like UnSegMedGAT.
- **Data Quality:** A number of files in the public dataset were corrupted or missing `z-spacing` metadata.
- **2D Slice-Based Analysis:** The model operates on 2D slices, which may not fully capture 3D volumetric context.

### Future Work
- **Acquire Weights for `Model_2`:** Enable a direct performance comparison with other architectures.
- **Explore Additional Pre-trained Models:** Survey literature for other publicly available models.
- **Aggregate 3D Explanations:** Combine 2D slice-level explanations to build a 3D understanding of model behavior.
