# Face Recognition Project

## Overview

This Jupyter Notebook demonstrates a face recognition pipeline using Principal Component Analysis (PCA) for eigenfaces, Linear Discriminant Analysis (LDA) for further dimensionality reduction, and a Multi-Layer Perceptron (MLP) classifier for prediction. The project initially attempts to load custom face images from a local directory but falls back to the Labeled Faces in the Wild (LFW) dataset from scikit-learn. It includes data loading, preprocessing, model training, evaluation, and visualization of results.

## Features

- Data loading from local directory or LFW dataset.
- Dimensionality reduction using PCA (eigenfaces) and LDA.
- Classification using MLPClassifier from scikit-learn.
- Evaluation with accuracy calculation.
- Visualization of predicted faces with true/predicted labels and probabilities.

## Dataset

The project uses the Labeled Faces in the Wild (LFW) dataset fetched via `sklearn.datasets.fetch_lfw_people` (minimum 70 faces per person, resized to 0.4 scale).

- **Total Samples**: Varies based on fetch (typically ~1288 images across 7 classes like Ariel Sharon, Colin Powell, etc.).
- **Image Size**: 50x37 pixels (after resizing).
- **Split**: 75% train, 25% test.

Note: The notebook attempts to load from a local path (`C:/Users/gaval/OneDrive/Desktop/faces`), but due to errors (e.g., skipping non-directory files), it defaults to LFW. Update the path for custom datasets.

## Requirements

### Dependencies

- Python 3.12.7 (based on notebook metadata).
- Libraries:
  - `opencv-python` (for image processing, though minimally used).
  - `matplotlib` (for plotting).
  - `sklearn` (for datasets, PCA, LDA, MLPClassifier, train_test_split).
  - `numpy` (for array operations).

Install dependencies using pip:

```bash
pip install opencv-python matplotlib scikit-learn numpy
```

### Environment

- Jupyter Notebook or compatible IDE (e.g., VS Code with Jupyter extension).
- No GPU required; runs on CPU.

## Installation

1. **Clone or Download the Notebook**:
   - Download `Face Recognition Project (1) (1).ipynb` from the source.

2. **Set Up Environment**:
   - Install dependencies as listed above.
   - If using a local dataset, ensure the directory structure is: `faces/person_name/image_files` (e.g., JPEGs).

3. **Download Dataset**:
   - The LFW dataset is automatically fetched via `fetch_lfw_people` (requires internet).

## Usage

### Running the Notebook

1. **Data Loading**:
   - Attempts local directory load; skips invalid files and uses LFW if none found.

2. **Preprocessing**:
   - Splits data into train/test.
   - Applies PCA (150 components) to extract eigenfaces.
   - Applies LDA on PCA-transformed data.

3. **Model Training**:
   - MLPClassifier: Hidden layers (512), activation 'relu', solver 'adam', max iterations 200, verbose=True.

4. **Prediction and Evaluation**:
   - Predicts on test set, calculates probabilities.
   - Computes accuracy (printed as "Final Accuracy").
   - Visualizes a gallery of test images with true/predicted labels.

5. **Customization**:
   - Update `dir_name` for custom images.
   - Adjust `n_components` in PCA or MLP hyperparameters for experimentation.

### Example Output

- Training logs show loss decreasing over iterations.
- Final accuracy printed (e.g., ~70-80% on LFW depending on run; actual value printed).
- Plotted gallery of 12 test faces with predictions.

## Results

- **Accuracy**: Calculated dynamically (e.g., ~70-80% on LFW depending on run; actual value printed).
- **Visualizations**: Eigenfaces (commented out) and prediction gallery.

## Contributing

- Fork and improve (e.g., add CNN-based recognition with TensorFlow, handle custom datasets better, or integrate real-time webcam detection).
- Issues: Report bugs or suggestions via the repository (if hosted on GitHub).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: Labeled Faces in the Wild (LFW) from scikit-learn.
- Libraries: scikit-learn team, OpenCV, Matplotlib.
- Inspired by classic face recognition techniques like Eigenfaces.

For questions, contact [your email or GitHub handle].
