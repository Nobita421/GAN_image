# Synthetic Image Generator — Vanilla GAN (Keras)

A complete implementation of a Vanilla GAN for synthetic image generation using TensorFlow/Keras, designed for privacy-preserving data augmentation.

## Project Structure

```
GAN_IMAGE/
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── data_loader.py       # Data loading and preprocessing
├── generator.py         # Generator network architecture
├── discriminator.py     # Discriminator network architecture
├── vanilla_gan.py       # VanillaGAN class (combines G & D)
├── train.py            # Training script
├── evaluation.py       # FID score evaluation
├── inference.py        # Generate synthetic images
├── app.py             # Streamlit web UI
├── smoke_test.py      # Setup and smoke tests
├── Dockerfile         # Docker configuration
└── utils/
    ├── visualizer.py  # Image grid visualization
    ├── metrics.py     # FID score calculation
    └── logger.py      # Logging utilities
```

## Quick Start

### 1. Setup

Run the installation script to install dependencies and verify setup:

```bash
install_and_test.bat
```

This will:
1. Install all required Python packages from `requirements.txt`
2. Run `smoke_test.py` to verify the installation and file structure

Alternatively, you can run these steps manually:

```bash
pip install -r requirements.txt
python smoke_test.py
```

### 3. Prepare Dataset

Create a data directory and add your images:

```bash
mkdir data\celeba_preprocessed
```

Place your training images (JPG/PNG) in `data\celeba_preprocessed\`. The data loader will automatically find and preprocess all images.

### 4. Train the GAN

```bash
python train.py
```

Training will:
- Load images from `dataset_path` in config.yaml
- Train for the specified number of epochs (default: 150)
- Save model checkpoints to `./checkpoints/`
- Generate sample images to `./samples/` every 5 epochs

### 5. Generate Synthetic Images

```bash
python inference.py
```

Or use the Streamlit UI:

```bash
streamlit run app.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
image_size: 64          # Image dimensions (64x64)
channels: 3             # RGB (3) or grayscale (1)
latent_dim: 100         # Size of latent noise vector
batch_size: 64          # Training batch size
learning_rate: 0.0002   # Adam learning rate
beta1: 0.5             # Adam beta1 parameter
epochs: 150            # Number of training epochs
sample_interval: 5     # Save samples every N epochs
dataset_path: ./data/celeba_preprocessed
save_dir: ./checkpoints
samples_dir: ./samples
```

## Module Descriptions

### Core Modules

- **generator.py**: Upsampling generator (4x4 → 8x8 → 16x16 → 32x32 → 64x64)
- **discriminator.py**: Convolutional discriminator with BatchNorm
- **vanilla_gan.py**: VanillaGAN class that combines G & D with Adam optimizers
- **data_loader.py**: Image preprocessing and TensorFlow dataset pipeline

### Training & Evaluation

- **train.py**: Main training loop with label smoothing
- **evaluation.py**: Calculate FID score on generated vs real images
- **inference.py**: Generate N synthetic images from trained model

### Utilities

- **utils/visualizer.py**: Create image grids for visualization
- **utils/metrics.py**: InceptionV3-based FID score calculation
- **utils/logger.py**: Logging setup

### UI & Deployment

- **app.py**: Interactive Streamlit web interface
- **Dockerfile**: Container for deployment

## Smoke Test Results

Run `python smoke_test.py` to verify:
- ✓ Generator architecture (parameters count)
- ✓ Discriminator architecture (parameters count)
- ✓ VanillaGAN compilation
- ✓ Config loading
- ✓ Utils modules import
- ✓ Inference module

## Training Tips

1. **Dataset Size**: GANs benefit from thousands of images. Use at least 1000+ images for good results.

2. **Monitoring**: Check `./samples/` directory during training to monitor generator progress.

3. **Checkpoints**: Models are saved to `./checkpoints/` every `sample_interval` epochs.

4. **Memory**: Adjust `batch_size` in config.yaml if you encounter OOM errors.

5. **Quality**: For better results, consider:
   - Using DCGAN architecture (add BatchNorm to generator)
   - Implementing WGAN-GP loss
   - Adding spectral normalization

## Docker Deployment

Build and run with Docker:

```bash
docker build -t gan-image-gen .
docker run -p 8501:8501 gan-image-gen
```

Access Streamlit UI at `http://localhost:8501`

## Privacy Considerations

When using for medical data (DICOM):
1. Add DICOM loader using `pydicom` in `data_loader.py`
2. Ensure all PHI (Protected Health Information) is removed
3. Comply with HIPAA and local regulations
4. Use secure storage for checkpoints and generated images

## Troubleshooting

**Issue**: Module import errors
- **Solution**: Run `python smoke_test.py` to diagnose

**Issue**: No dataset found
- **Solution**: Verify `dataset_path` in config.yaml points to directory with images

**Issue**: CUDA/GPU errors
- **Solution**: TensorFlow will auto-fallback to CPU. For GPU, ensure CUDA is installed.

**Issue**: Training very slow
- **Solution**: Reduce `batch_size` or `image_size` in config.yaml

## Next Steps

After basic setup:
1. ✅ Run smoke tests
2. ✅ Install dependencies
3. ⬜ Add training data
4. ⬜ Run training for 1 epoch to verify
5. ⬜ Monitor sample quality
6. ⬜ Evaluate with FID score
7. ⬜ Generate synthetic images

## License

See project documentation for license information.
