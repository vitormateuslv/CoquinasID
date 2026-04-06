# CoquinasID

Automated recognition of taphonomic patterns in μCT images of coquinas using computer vision.

## Overview

CoquinasID is a Python-based workflow for automated recognition and quantification of taphonomic patterns in μCT images.

The algorithm performs:

- Object detection using watershed and morphological operations  
- Classification of shells, fragments, and sand  
- Angle classification (concordant, oblique, vertical)  
- Concavity classification (up/down)  
- Morphometric measurements (length and thickness in mm)  

## Installation

Install required libraries:

```bash
pip install opencv-python numpy matplotlib scipy

```

## Usage

The algorithm is available in two formats:

- Jupyter Notebook (`.ipynb`)  
- Python script (`.py`)  

### Running in Jupyter Notebook or VS Code

1. Open the notebook:
   coquinasid.ipynb

2. Run all cells sequentially

3. Insert the path to the input image when requested

4. The outputs (figures and metrics) will be generated automatically

### Running in Google Colab (recommended)

1. Open Google Colab  
2. Upload or paste the notebook code  
3. Run all cells  
4. Provide the image path (or upload the image directly)

No additional configuration is required.

## Input Data

- Grayscale μCT images  
- Recommended format: `.png`, `.jpg`, or `.tif`

## Outputs

- Classified images  
- Pie charts  
- Morphometric measurements  
- Execution time  

## Notes

The workflow was designed for ease of use, requiring minimal user interaction.  
All analyses are performed automatically after providing the input image.

## Author

Vitor Mateus Lopes Vargas
