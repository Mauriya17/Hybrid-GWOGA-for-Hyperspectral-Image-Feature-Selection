\# Hybrid GWOGA for Image Feature Selection



A hybrid Grey Wolf Optimizer–Genetic Algorithm (GWOGA) framework for feature selection and image classification using deep learning-based feature extraction and Support Vector Machines (SVM).



\## Overview



This project combines deep feature extraction with metaheuristic feature selection and SVM classification.



The overall pipeline is:



Image Dataset

→ ConvNeXt Feature Extraction

→ 512-D Deep Features

→ Grey Wolf Optimization (GWO)

→ Genetic Algorithm (GA)

→ Selected Features

→ SVM Classification

→ Final Evaluation



The objective is to reduce the number of features while maintaining strong classification performance.



\## Methodology



The framework consists of the following major stages:



1\. Dataset loading and preprocessing

2\. ConvNeXt-based deep feature extraction

3\. Feature scaling using StandardScaler

4\. Feature selection using Grey Wolf Optimization (GWO)

5\. Genetic Algorithm (GA) refinement using the GWO solution

6\. SVM classification using the selected features

7\. Evaluation on a held-out test set



\### Deep Feature Extraction



ConvNeXtBase is used as the deep feature extractor.



The model uses an image input size of 224 × 224 and produces a 512-dimensional feature representation after the classification head.



The feature extractor is fine-tuned before the final feature extraction stage.



\### Grey Wolf Optimization



GWO searches for a subset of useful features.



Each candidate solution represents a binary feature-selection vector.



The fitness function considers both:



\- Classification error

\- Number of selected features



The objective is therefore to obtain a feature subset that provides good classification performance while reducing the feature dimension.



\### Genetic Algorithm



The Genetic Algorithm further searches for an effective feature subset.



The best solution obtained from GWO is injected into the initial GA population. The GA then applies selection, crossover, and mutation to continue the search.



\### Classification



An RBF-kernel Support Vector Machine (SVM) is trained using the selected features.



The final model package contains:



\- Trained SVM model

\- Feature scaler

\- Label encoder

\- Selected feature indices



\## Datasets



The project currently supports the following datasets:



\### Medical Waste 4.0



Dataset:

https://zenodo.org/records/7643417



\### X-ray Bone Fracture Dataset



Dataset:

https://data.mendeley.com/datasets/8d9kn57pdj/1



\### UC Merced Land Use Dataset



Dataset:

https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset



The datasets are not included in this repository.



Please download the datasets from their respective sources and place the required ZIP files inside the `datasets/` directory.



\## Project Structure



```text

Hybrid-GWOGA-for-Hyperspectral-Image-Feature-Selection/

│

├── main.py

├── baseline\_model.py

├── README.md

├── requirements.txt

├── .gitignore

│

├── datasets/

│   └── Dataset ZIP files

│

├── project\_output/

│   └── Generated results and model files

│

└── docs/

&#x20;   └── Reference paper information

