Installation Guide
=================

Requirements
-----------

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

Installation Steps
----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/Computational-Imaging-LAB/T2_FoundationModel.git
      cd T2_FoundationModel

2. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Verify installation:

   .. code-block:: bash

      python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
      python -c "print('CUDA available:', torch.cuda.is_available())"
