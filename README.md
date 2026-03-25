# Spectra Analysis: Paint and Non-Paint Microplastic Classification

## Overview

This repoository accomapnies the forthcoming paper **A Paint Library Of Plastic Particles (PLOPP): Fourier transform infrared spectral analysis of paint microplastics**, Zoie T. Diana, Madeleine Milne, Jacob Ford, Ron Rubinovitz, Andrew Turner, Chelsea M. Rochman.

This project aims to address the challenge of characterizing paint microplastics in the environment and differentiating them from non-paint microplastics. We have created a comprehensive Fourier transform infrared spectroscopy (FTIR) library named the Paint Library of Plastic Products (PLOPP), which includes 263 spectra from 90 different paints used in various sectors.

To enhance the identification of paint microplastics, we utilized machine learning (ML) techniques to classify spectral data as either paint or non-paint microplastics. This document provides an overview of the ML component implemented in this project.

## Machine Learning Component

### Objective

The primary objective of the ML component is to train a model that can accurately classify FTIR spectra as either paint microplastics or non-paint microplastics. This classification aids in the verification of environmental samples and supports the broader goal of understanding microplastic pollution sources.

### Data

Training Data: The PLOPP library consisting of 263 spectra from 90 paints.Test Data: Environmental microplastic samples collected from the River Thames, United Kingdom.

## Methodology

1.  Data Preprocessing:
    1.  Spectral data normalization and transformation.
    2.  Feature extraction from the spectra.
2.  Model Selection:
    1.  A Random Forest is used for its effectiveness in handling complex datasets and providing high accuracy.
3.  Pipeline Creation:
    1.  A Scikit-learn Pipeline is used to streamline the preprocessing and model training steps.

## Evaluation

-   Accuracy: The model's accuracy is evaluated using cross-validation on the training set.
-   Prediction: The model predicts the classification of environmental microplastic samples.

## Results

The trained model successfully classifies the spectra, enabling the differentiation between paint and non-paint microplastics in environmental samples. This classification supports the utility of the PLOPP library in environmental microplastic research.

## Reproducible Setup (Windows and macOS)

For repeatable environments across PCs and Macs, use the script pipeline in
`analysis/script` with `uv` and the committed lockfile:

1. Install `uv`.
2. Ensure Python `3.13.7` is available.
3. From `analysis/script`, run:

```bash
uv sync
uv run python main.py
```

`analysis/script/pyproject.toml` + `analysis/script/uv.lock` are the canonical
environment definition for reproducible runs.

## Notebook Setup (analysis/src)

If your priority is running `analysis/src/spectra_analysis.ipynb`, use the
notebook setup scripts:

- Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File analysis/src/setup_notebook.ps1
```

- macOS/Linux:

```bash
bash analysis/src/setup_notebook.sh
```

Then open `analysis/src/spectra_analysis.ipynb` in VS Code and choose kernel:
`PLOPP Notebook (.venv)`. 