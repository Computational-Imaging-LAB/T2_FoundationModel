# T2 Foundation Model

A Vision Transformer-based Masked Autoencoder Variational Autoencoder (MAE-VAE) for medical image analysis, specifically designed for T2-weighted MRI processing.

## Features

- Vision Transformer (ViT) based architecture
- Masked autoencoding with variational inference
- NIfTI file support for medical imaging
- Slice-wise and volume-wise processing
- Advanced visualization tools

## Installation

```bash
git clone https://github.com/Computational-Imaging-LAB/T2_FoundationModel.git
cd T2_FoundationModel
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --train_dir /path/to/train/nifti --val_dir /path/to/val/nifti
```

### Inference

```bash
python inference.py --nifti_path /path/to/nifti/file.nii.gz
```

## Documentation

Documentation is available at `docs/build/html/index.html`. To build the documentation:

```bash
cd docs
make html
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{t2_foundation_model,
  title = {T2 Foundation Model},
  author = {Computational Imaging Lab},
  year = {2025},
  url = {https://github.com/Computational-Imaging-LAB/T2_FoundationModel}
}
