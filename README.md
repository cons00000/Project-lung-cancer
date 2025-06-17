# Explainable AI for Lung Cancer Stage Classification

<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.8+-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="Status" src="https://img.shields.io/badge/status-in%20progress-yellow.svg">
</p>

This project applies and analyzes a pre-trained machine learning model for lung cancer stage classification, with a strong emphasis on transparency through Explainable AI (XAI) techniques. By implementing post-hoc explanation methods on a state-of-the-art model, we gain crucial insights into its decision-making process, evaluating its clinical relevance and reliability.

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Project Pipeline](#project-pipeline)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation and Data Setup](#installation-and-data-setup)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Data Exploration: TNM Staging](#data-exploration-tnm-staging)
- [Model Analysis and Explainability](#model-analysis-and-explainability)
  - [Segmentation Results](#segmentation-results)
  - [Explainable AI (XAI) Analysis](#explainable-ai-xai-analysis)
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
- **End-to-End Inference Pipeline:** From data preprocessing and segmentation to prediction and explanation using pre-trained models.
- **State-of-the-Art Model Application:** Implements the **DuneAI** model for segmentation and classification. An implementation for a second model (**UnSegMedGAT**) is included, though its pre-trained weights were not available.
- **Deep Explainability:** Leverages a wide array of XAI techniques (Saliency, Integrated Gradients, Grad-CAM) to provide a holistic view of the model's decision-making.
- **Clinical Focus:** Grounds the analysis in established medical standards like TNM staging to assess practical utility.

## Project Pipeline

Our methodology is an inference and analysis pipeline focused on a pre-trained model:

**`Data Acquisition`** → **`Preprocessing & Segmentation`** → **`Prediction (Inference)`** → **`Explainability Analysis`**

1.  **Inference Phase:** We process raw medical imaging data, use the pre-trained model to segment regions of interest (lungs and tumors), and generate stage classifications.
2.  **Explainability Phase:** We apply post-hoc XAI methods to the model's predictions to generate visual and quantitative explanations, linking its internal logic back to clinical factors.

## Getting Started

Follow these steps to set up the project environment and download the necessary data.

### Prerequisites
- **Python:** 3.8+
- **Git:** [https://git-scm.com/](https://git-scm.com/)
- **Conda:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

### Installation and Data Setup

**Step 1: Clone the Repository**
```bash
git clone https://github.com/cons000000/project-lung-cancer.git
cd project-lung-cancer
```

**Step 2: Set Up Conda Environments**

This project requires two separate environments due to conflicting dependencies.

*   **Environment 1: `lung-dataprep`**
    ```bash
    conda create --name lung-dataprep python=3.8 -y
    conda activate lung-dataprep
    pip install -r Model_1/Data_preparation/requirements.txt
    ```

*   **Environment 2: `lung-segment`**
    ```bash
    conda create --name lung-segment python=3.10 -y
    conda activate lung-segment
    pip install -r Model_1/Segmentation/requirements.txt
    ```

**Step 3: Download the Dataset**

This project uses the **Lung-PET-CT-Dx** dataset from The Cancer Imaging Archive (TCIA).

> Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). *A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx) [Data set]*. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2020.NNC2-0461

1.  Go to the [TCIA Data Access Page](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/).
2.  Create a TCIA account and accept the data usage agreement.
3.  Download both **"Images"** and **"Clinical Data"**.
4.  Place all downloaded data into a directory named `NIH dataset_raw/` at the root of the project.

## Repository Structure
The repository is organized to separate pipeline stages and model implementations.

```
project-lung-cancer/
├── NIH dataset_raw/              # Raw data (place downloaded files here)
├── Data-analysis/                # Jupyter notebook for initial data exploration
├── Model_1/                      # Primary model approach (DuneAI)
│   ├── Data_preparation/         # Scripts for data cleaning and feature engineering
│   ├── Segmentation/             # Lung and tumor segmentation using DuneAI
│   ├── T_Stage_Classification/   # Calculation of RECIST from segmentation masks
│   └── Xai/                      # Notebooks for generating XAI explanations
├── Model_2/                      # Second approach (UnSegMedGAT, weights unavailable)
└── Visualize_lung_mask/          # Notebook for visualizing segmentation masks
├── README.md
└── ... (other config files)
```

## Usage
To run the analysis, navigate to the relevant directories and launch the Jupyter notebooks. Remember to activate the correct Conda environment for each task.

- For **data preparation**: `conda activate lung-dataprep`
- For **segmentation and XAI**: `conda activate lung-segment`

**Example:** To run the main segmentation notebook:
```bash
conda activate lung-segment
cd Model_1/Segmentation/
jupyter notebook Segmentation.ipynb
```

## Data Exploration: TNM Staging

Our analysis is grounded in the clinical **TNM Staging System**, which describes the anatomical extent of cancer.

| Aspect                 | TNM Staging                               | Histopathological Grading                  |
| ---------------------- | ----------------------------------------- | ------------------------------------------ |
| **Focus**              | Anatomical spread of the tumor            | Cellular appearance & aggressiveness       |
| **Components**         | **T** (Tumor size), **N** (Nodes), **M** (Metastasis) | Differentiation (Grade G1-G3)              |

### Dataset Distributions
Below are the distributions of T, N, and M stages across the dataset.

<p align="center">
  <img src="chart2.svg" alt="T-Stage distribution" width="30%"/>
  <img src="chart1.svg" alt="N-Stage distribution" width="30%"/>
  <img src="chart3.svg" alt="M-Stage distribution" width="30%"/>
</p>
<p align="center">
  <b>Fig 1.</b> T-Stage Distribution &nbsp;&nbsp;&nbsp;&nbsp; <b>Fig 2.</b> N-Stage Distribution &nbsp;&nbsp;&nbsp;&nbsp; <b>Fig 3.</b> M-Stage Distribution
</p>

## Model Analysis and Explainability

### Segmentation Results
We utilized **DuneAI**, a pre-trained deep learning model for automated segmentation of non-small cell lung cancer (NSCLC). This model serves as the core of our analysis pipeline.

- **DuneAI Paper:** Primakov, S.P., et al. *Automated detection and segmentation of non-small cell lung cancer computed tomography images.* Nat Commun 13, 3423 (2022).
- **Toolbox Used:** [precision-medicine-toolbox](https://github.com/primakov/precision-medicine-toolbox)

<p align="center">
  <img src="output.png" alt="Model segmentation" width="60%"/>
  <br>
  <b>Figure 4:</b> DuneAI segmentation results on a sample patient's middle CT slice.
</p>

### Explainable AI (XAI) Analysis

To understand *how* the pre-trained model makes its decisions, we used the [Xplique](https://github.com/deel-ai/xplique) library to generate attribution maps. These maps show which pixels in a given 2D slice were most influential for the model's prediction.

| Method                 | Description                                                                 | How to Interpret                                                                |
| ---------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Saliency**           | Highlights pixels with the strongest influence on the output via gradients. | **Brighter regions are most important.** Shows raw model sensitivity.           |
| **Integrated Gradients** | Aggregates gradients along a path from a baseline (black image) to the input. | Provides a more stable and reliable attribution than simple gradients.       |
| **SmoothGrad**         | Averages gradients over multiple noisy versions of the input image.         | **Reduces noise** to reveal the true underlying patterns the model focuses on.  |
| **Sobol Attribution**  | Uses sensitivity analysis to measure feature contributions and interactions. | Reveals how different image regions **work together** to influence a decision. |

<p align="center">
  <img src="saliencyoutput.png" alt="Saliency explanation" width="45%"/>
  <img src="integratedgradientsoutput.png" alt="Integrated gradients explanation" width="45%"/>
  <br>
  <b>Figure 5:</b> Saliency (left) vs. Integrated Gradients (right) on the same slice.
</p>

## Performance Evaluation

The pre-trained model's classification performance was evaluated using standard metrics. The confusion matrix provides a detailed breakdown of correct and incorrect predictions.

<p align="center">
  <img src="confusionmatrixoutput.png" alt="Confusion Matrix" width="45%"/>
  <img src="distributionerrorsoutput.png" alt="Error Distribution" width="45%"/>
  <br>
  <b>Figure 6:</b> Confusion Matrix (left) and Distribution of Prediction Errors (right).
</p>

## Limitations and Future Work

### Known Issues
- **Limited Model Comparison:** The project initially aimed to compare several pre-trained models. However, only the DuneAI model was fully implementable due to the unavailability of public weights for other models like UnSegMedGAT.
- **Corrupted Data:** A number of files in the public dataset were corrupted and had to be excluded.
- **Missing Metadata:** Several files were missing `z-spacing` metadata. A default value of `1.0 mm` was applied, which may impact the accuracy of 3D volumetric measurements.

### Future Work
- **Acquire Weights for `Model_2`:** Obtain the necessary weights for the UnSegMedGAT model to enable a direct performance comparison.
- **Explore Additional Pre-trained Models:** Survey literature for other publicly available models relevant to lung cancer analysis.
- **Improve 3D Context:** While the model operates on 2D slices, future work could involve aggregating slice-level explanations to build a 3D understanding of model behavior.
- **Deploy as an Analysis Tool:** Package the inference and XAI pipeline into an interactive tool for researchers to analyze other pre-trained models.

## Citation

If you use this work, please cite the original dataset and consider citing this repository.

```bibtex
@misc{ProjectLungCancerXAI2023,
  author = {Your Name/Team Name},
  title = {Project-Lung-Cancer: Explainable AI for Lung Cancer Stage Classification},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cons000000/project-lung-cancer}}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.