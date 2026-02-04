# Setup Guide for Jute Pest Classification

## Quick Start

1. Clone or download this repository

2. (Recommended) Create and activate a virtual environment. **For GPU:** use **Python 3.10**:
   ```bash
   py -3.10 -m venv .venv
   .\.venv\Scripts\activate
   ```
   For CPU-only you can use Python 3.11 or 3.12.

3. **GPU (optional):** Install CUDA 11.2 and cuDNN 8.1 and add them to PATH so TensorFlow can use your NVIDIA GPU. Without these, the notebook runs on CPU.

4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

5. Organize your dataset:
   
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

6. **Run the whole project** (from the project root `jute-pest-classification`):
   ```bash
   python -m jupyter notebook notebooks/jute_pest_classification.ipynb
   ```
   Or with Jupyter Lab:
   ```bash
   python -m jupyter lab notebooks/jute_pest_classification.ipynb
   ```
   If `jupyter` is not recognized, always use `python -m jupyter` instead of `jupyter`.
   In the notebook, use **Cell → Run All** to run the full pipeline (load data, train models, evaluate, save results).

   Alternatively, run the notebook from the command line without opening the browser:
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/jute_pest_classification.ipynb
   ```

## Dataset Organization

The dataset should be organized with:
- 17 pest classes as subdirectories
- train/, val/, and test/ splits
- Images in JPG/JPEG/PNG format

## Expected Dataset Size

- Training: ~5,161 images
- Validation: ~1,282 images  
- Test: 379 images
- Total: ~6,822 images

## GPU Optimization (Important)

**The project does not optimize or use your GPU unless it is run under specific settings.** These are the settings that work with this setup (tested configuration):

- **Python 3.10** (TensorFlow 2.10 does not support GPU on Python 3.11+)
- **CUDA 11.2** and **cuDNN 8.1** installed and on PATH (required for this GPU/TensorFlow combo)
- Virtual environment created with Python 3.10 (e.g. `py -3.10 -m venv .venv`)
- Dependencies installed in that same environment (`pip install -r requirements.txt`)

Without these, the notebook will run on CPU only. Different GPUs may require different CUDA/cuDNN versions—check your GPU’s compatibility.

## Hardware Requirements

### Minimum Requirements:
- 8GB RAM
- CPU (slower training)

### Recommended (GPU):
- 16GB+ RAM
- **NVIDIA GPU** with **CUDA 11.2** and **cuDNN 8.1** (TensorFlow 2.10)

## Training Time Estimates

- ResNet50: ~2-4 hours (GPU), ~8-12 hours (CPU)
- ResNet101: ~3-5 hours (GPU), ~12-18 hours (CPU)
- EfficientNetB0: ~2-3 hours (GPU), ~6-10 hours (CPU)
- VGG16: ~3-4 hours (GPU), ~10-15 hours (CPU)
- DenseNet201: ~4-6 hours (GPU), ~15-20 hours (CPU)

Note: Times are approximate and depend on hardware configuration.

## Troubleshooting

### Issue: "Could not find dataset"
**Solution:** Ensure the dataset is located in `data/Jute_Pest_Dataset/` relative to the notebook location.

### Issue: Out of memory errors
**Solution:** 
- Reduce batch size in the notebook 
- Use a smaller model first (e.g., EfficientNetB0 instead of DenseNet201)

### Issue: GPU not detected (Windows)
**Solution:** For GPU use **Python 3.10**, **TensorFlow 2.10.0**, **CUDA 11.2**, **cuDNN 8.1**. Create an env with Python 3.10, install CUDA 11.2 and cuDNN 8.1, then `pip install -r requirements.txt`. Verify: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` should list your GPU.

### Issue: CUDA out of memory (NVIDIA GPU)
**Solution:**
- Reduce batch size
- Use mixed precision training
- Close other GPU-intensive applications

## Additional Notes

- The notebook automatically creates train/val split (80/20) from the training directory
- Models are saved automatically during training (best model based on validation accuracy)
- All visualizations are saved as PNG files in the results/ directory
- Results are saved to `results/model_comparison_results.csv`
