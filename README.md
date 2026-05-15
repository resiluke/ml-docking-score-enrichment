# Machine Learning for Docking Score Prediction and Dataset Enrichment

This repository contains the source codes developed for a thesis focused on machine learning prediction of docking scores and systematic enrichment of training datasets in virtual screening workflows.


---

## Repository Structure

```text
scripts/
├── 01_dataset_preparation/
├── 02_dataset_characterization/
├── 03_hyperparameter_optimization/
├── 04_model_training/
├── 05_evaluation/
└── 06_visualization/ 

```


## Main Dependencies

- Python 
- TensorFlow
- DScribe
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Hyperopt

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Implemented Molecular Descriptors

- SOAP (Smooth Overlap of Atomic Positions)
- ACSF (Atom-Centered Symmetry Functions)
- MBTR (Many-Body Tensor Representation)
- Coulomb Matrix
Hyperparameter optimization using Hyperopt
---

## Machine Learning Workflow

1. Dataset preparation and cross-validation split generation
2. Descriptor calculation
3. Neural network training using TensorFlow/Keras
4. Model evaluation on in-vitro datasets
5. Visualization of classification and regression performance


---

## Datasets

The original datasets are not included in this repository due to file size limitations.

The datasets contain molecular structures in XYZ format together with docking scores.

### Dataset source

Bucinsky, L., et al.  
*Data for: Advances and critical assessment of machine learning techniques for prediction of docking scores.*  
Dryad Digital Repository, 2023.  
https://doi.org/10.5061/dryad.zgmsbccg7

---

## Outputs

The scripts generate:

- ROC and PR curves
- Confusion matrix statistics
- MSE analyses
- Density quadrant plots
- Recall vs precision visualizations
- Model comparison plots
- Hyperparameter optimization summaries

---
