.. T2 Foundation Model documentation master file, created by
   sphinx-quickstart on Sat Jan 25 17:17:12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

T2 Foundation Model Documentation
================================

A Vision Transformer-based Masked Autoencoder Variational Autoencoder (MAE-VAE) for medical image analysis, specifically designed for T2-weighted MRI processing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   modules/mae_vae
   modules/nifti_dataset
   modules/inference
   modules/train

Installation
------------

.. code-block:: bash

   git clone https://github.com/Computational-Imaging-LAB/T2_FoundationModel.git
   cd T2_FoundationModel
   pip install -r requirements.txt

Quick Start
----------

Training
^^^^^^^^

.. code-block:: python

   python train.py --train_dir /path/to/train/nifti --val_dir /path/to/val/nifti

Inference
^^^^^^^^

.. code-block:: python

   python inference.py --nifti_path /path/to/nifti/file.nii.gz

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
