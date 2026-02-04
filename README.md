# Jute Pest Classification using Deep Learning

A deep learning project for classifying 17 different types of jute pests using transfer learning and convolutional neural networks. This project compares multiple pre-trained architectures (ResNet50, ResNet101, EfficientNetB0, VGG16, DenseNet201) to identify the optimal model for agricultural pest classification.

## Project Overview

This project addresses a real-world agricultural problem: identifying jute pests from images. Jute is an important fiber crop, and early pest identification is crucial for effective pest management. The model uses transfer learning to leverage pre-trained ImageNet weights and fine-tunes them for pest classification.

**Key Features:**
- Multi-class classification (17 pest types)
- Transfer learning with 5 different architectures
- Comprehensive data augmentation
- Model comparison and evaluation
- Confusion matrix visualization
- Training history analysis

## Dataset

The dataset contains images of 17 different jute pest types:
- Beet Armyworm
- Black Hairy
- Cutworm
- Field Cricket
- Jute Aphid
- Jute Hairy
- Jute Red Mite
- Jute Semilooper
- Jute Stem Girdler
- Jute Stem Weevil
- Leaf Beetle
- Mealybug
- Pod Borer
- Scopula Emissaria
- Termite
- Termite odontotermes (Rambur)
- Yellow Mite

**Dataset Statistics:**
- Training: ~5,161 images
- Validation: ~1,282 images
- Test: 379 images
- Total: ~6,822 images

## Architecture

The project uses transfer learning with the following architectures:
1. **ResNet50** - 50-layer residual network
2. **ResNet101** - 101-layer residual network
3. **EfficientNetB0** - Efficient and accurate CNN
4. **VGG16** - 16-layer VGG network
5. **DenseNet201** - Densely connected CNN

Each model follows this structure:
- Pre-trained base model (ImageNet weights, frozen)
- Global Average Pooling
- Batch Normalization
- Dropout (0.2)
- Dense layer (256 units, ReLU, L2 regularization)
- Output layer (17 units, softmax)

## Installation

### Prerequisites
- **Python 3.10** recommended for GPU
- **TensorFlow 2.10.0** (CUDA 11.2, cuDNN 8.1 for NVIDIA GPU)
- CUDA-compatible GPU (recommended)

> **Important:** The project does **not** use your GPU unless run with the right setup. For this configuration (tested): Python 3.10, CUDA 11.2, cuDNN 8.1. Different GPUs may need different versions. See [SETUP.md](SETUP.md#gpu-optimization-important) for details.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/angelaykang/jute-pest-classification.git
cd jute-pest-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Organize your dataset:
   
   Place your dataset in the following structure:
   ```
   jute-pest-classification/
   ├── data/
   │   └── Jute_Pest_Dataset/
   │       ├── train/
   │       │   ├── Beet Armyworm/
   │       │   ├── Black Hairy/
   │       │   └── ... (all 17 classes)
   │       ├── val/
   │       │   ├── Beet Armyworm/
   │       │   └── ... (all 17 classes)
   │       └── test/
   │           ├── Beet Armyworm/
   │           └── ... (all 17 classes)
   ```

## Usage

### Running the Notebook

1. Open the main notebook in Jupyter Notebook or Jupyter Lab:
   - `notebooks/jute_pest_classification.ipynb`

2. Update the dataset path in the notebook if necessary:
   ```python
   path = Path('../data/Jute_Pest_Dataset')
   ```

3. Execute all cells to:
   - Load and preprocess the dataset
   - Train multiple models
   - Evaluate and compare models
   - Generate visualizations

### Training a Single Model

The notebook includes functions to build and train models:

```python
from tensorflow.keras.applications import ResNet50

model = build_model(ResNet50, 'ResNet50', n_classes=17, dr=0.2, l2=0.01)
model = setup(model, lr=0.001)
history = fit(model, train_loader, val_loader, epochs=50, name='ResNet50')
```

### Evaluating Models

```python
results = eval(model, test_loader, name='ResNet50')
plot_cm(true_labels, predictions, classes, 'ResNet50', 'Test')
```

## Results

The project compares multiple models and saves results to `results/model_comparison_results.csv`. The following metrics are evaluated:
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1 Score (weighted average)
- AUC (one-vs-rest)

### Model Performance Summary (Test Set)

From `results/model_comparison_results.csv`:

| Model | Test Accuracy | F1 Score | AUC |
|-------|---------------|----------|-----|
| **DenseNet201** | **96.6%** | **0.97** | **0.999** |
| VGG16 | 83.1% | 0.82 | 0.986 |
| ResNet50 | 25.3% | 0.21 | 0.820 |
| ResNet101 | 25.1% | 0.25 | 0.779 |
| EfficientNetB0 | 8.4% | 0.02 | 0.592 |

### Best Model

Based on the evaluation, **DenseNet201** achieved the best performance with 96.6% test accuracy.

**Note:** ResNet50, ResNet101, and EfficientNetB0 showed poor performance on this dataset. This may be due to the frozen backbone approach not being optimal for these architectures on insect images—fine-tuning the backbone layers or using different learning rates could improve results.

## Project Structure

```
jute-pest-classification/
├── README.md                              # project documentation
├── SETUP.md                               # setup guide
├── requirements.txt                       # dependencies
├── .gitignore                             # git ignore rules
├── notebooks/
│   └── jute_pest_classification.ipynb
├── results/
│   ├── model_comparison_results.csv       # model comparison results
│   ├── all_models_comparison.png          # overall comparison visualization
│   ├── augmented_images.png               # augmentation examples
│   ├── sample_images.png                  # sample images per class
│   ├── ResNet50_training_history.png      # per-model training curves
│   ├── ResNet101_training_history.png
│   ├── EfficientNetB0_training_history.png
│   ├── VGG16_training_history.png
│   ├── DenseNet201_training_history.png
│   ├── ResNet50_Test_confusion_matrix.png # per-model confusion matrices
│   ├── ResNet101_Test_confusion_matrix.png
│   ├── EfficientNetB0_Test_confusion_matrix.png
│   ├── VGG16_Test_confusion_matrix.png
│   └── DenseNet201_Test_confusion_matrix.png
└── data/                                  # dataset directory (not in repo)
    └── Jute_Pest_Dataset/
        ├── train/
        ├── val/
        └── test/
```

## Data Augmentation

The training data is augmented with the following transformations:
- Rotation (±20 degrees)
- Width/height shifts (±20%)
- Shear transformation (±20%)
- Zoom (±20%)
- Horizontal and vertical flips
- Brightness adjustment (0.8-1.2x)
- Random crop and contrast adjustment

## Key Techniques

1. **Transfer Learning**: Leverages pre-trained ImageNet weights
2. **Data Augmentation**: Increases dataset diversity and reduces overfitting
3. **Early Stopping**: Prevents overfitting during training
4. **Model Checkpointing**: Saves best model based on validation accuracy
5. **Regularization**: L2 regularization and dropout to prevent overfitting
6. **Class Imbalance Handling**: Weighted metrics for evaluation
7. **GPU optimization**: Memory growth and mixed precision (float16) when a GPU is available; LossScaleOptimizer for stable training

## Model Comparison

The project trains and compares 5 different architectures:
- All models use the same training configuration
- Models are evaluated on train, validation, and test sets
- Results are saved for comparison

## Visualizations

The notebook generates several visualizations saved in the `results/` directory:
- Sample images from each pest class
- Augmented training images
- Training history (loss and accuracy curves)
- Confusion matrices for each model
- Model comparison charts

## Technical Details

- **Image Size**: 224x224 pixels
- **Batch Size**: 64 
- **Learning Rate**: 0.001 (Adam optimizer)
- **Epochs**: 50 (with early stopping)
- **Dropout Rate**: 0.2
- **L2 Regularization**: 0.01
- **Train/Val Split**: 80/20

## Dependencies

See `requirements.txt` for the complete list. Key dependencies include:
- TensorFlow 2.10.0
- NumPy 1.24.3
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Scikit-learn

## Acknowledgments

- Dataset: Jute Pest Dataset
- Pre-trained models: ImageNet weights from TensorFlow/Keras

## Additional Information

- The dataset is large (~7,000 images) and is not included in this repository
- Git LFS can be used for large files if version control of the dataset is required
- Training time varies by model and hardware (GPU recommended)
- The notebook is designed to run on CPU or GPU (NVIDIA with CUDA)
