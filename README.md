# Project-Lung-Cancer: Explainable AI for Lung Cancer Stage Classification

## Introduction
This project aims to develop and analyze machine learning models for lung cancer stage classification using Explainable AI (XAI) techniques. By implementing post hoc explanation methods, we gain insights into how our models make decisions, enhancing both model performance and clinical interpretability.

### Project Objectives
- Develop accurate classification models for lung cancer staging
- Apply post hoc XAI methods to understand model decision-making processes
- Create a framework for medical image analysis with transparent, interpretable results
- Advance skills in machine learning and explainable AI techniques

### Dataset
This project utilizes the Lung-PET-CT-Dx dataset:
> Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2020.NNC2-0461

The dataset includes CT and PET/CT scans from lung cancer patients with corresponding clinical information, providing a comprehensive basis for cancer staging analysis.

## Methodology
Our approach consists of two primary phases:

1. **Classification**: Developing models to classify patient data according to cancer stage
   - Data preprocessing and feature extraction
   - Model selection and optimization
   - Performance evaluation using clinical metrics

2. **Explainability**: Implementing post hoc XAI methods to understand model decisions
   - Feature importance analysis
   - Visual explanations of model predictions
   - Correlation between model features and clinical factors

## Installation and Setup

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Getting Started
1. Clone this repository
```bash
git clone https://github.com/cons000000/project-lung-cancer.git
cd project-lung-cancer
```
2. Download the dataset in the [Data Access](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) section, both the files names "Images" and "Clinical Data" are needed.

## Data Access
The Lung-PET-CT-Dx dataset is available through The Cancer Imaging Archive (TCIA). To access the data:

1. Visit [TCIA's collection page](https://www.cancerimagingarchive.net/)
2. Create an account and accept the data usage agreement
3. Download the dataset using the NBIA Data Retriever tool
4. Place the downloaded data in the `NIH dataset_raw/` directory

## Repository Structure
```
project-lung-cancer/
├── NIH dataset_raw/                       # Data directory (not tracked by git)
│   ├── manifest-1608669183333/            # Raw dataset files
│   ├── NRRD/                     # Dataset files converted to NRRD
│   └── Processed/                # NNRD files + mask of the tumor
├── src/
│   ├── Data_preparation/       # Data processing scripts
│   │   ├── AnalysisBox.py      # Analysis utilities
│   │   ├── DataSet.py          # Dataset class definitions
│   │   ├── FeaturesSet.py      # Feature extraction
│   │   ├── GenerateResultBox.py # Result generation utilities
│   │   ├── ResultSet.py        # Result processing
│   │   ├── ToolBox.py          # General utilities
│   │   ├── data_preparation.ipynb # Data preparation notebook
│   │   └── requirements.txt    # Dependencies for data preparation
│   ├── Segmentation/           # Image segmentation
│   │   ├── model_files/        # Pretrained model weights
│   │   ├── Generator_v1.py     # Data generator for segmentation
│   │   ├── lung_extraction_funcs_13_09.py # Lung extraction utilities
│   │   ├── Segmentation.ipynb  # Segmentation workflow notebook
│   │   ├── TheDuneAI.py        # Core AI model implementation
│   │   └── visualize_data.ipynb # Visualization for segmentation quality control
└── README.md                   # Project description
```

## Usage
1. **Data Preparation**: Run the data preparation notebook to process the raw CT and PET/CT scans
```bash
jupyter notebook src/Data_preparation/data_preparation.ipynb
```

2. **Segmentation**: Run the segmentation notebook to extract lung regions and identify tumors
```bash
jupyter notebook src/Segmentation/Segmentation.ipynb
```

## Results


## Acknowledgments
- The Cancer Imaging Archive (TCIA) for providing the dataset
- [List any other acknowledgments here]