# CSI Project Documentation

## Project Overview
This project aims to perform deep learning modeling using CSI (Channel State Information) data. The primary research directions include Attention mechanisms, Generative Adversarial Networks (GAN), Identity Recognition, and Intruder Detection. The project contains multiple research modules, each with corresponding code for data preprocessing, model construction, training, and evaluation.

## Directory Structure
```
Research1/                # Research Direction 1: Attention and GAN
  Attention/              # Attention mechanism model training results
  GAN/                    # GAN model training results
  dataloader_ATT.py       # Data loader for the attention mechanism
  loss_ATT.py             # Loss function for the attention mechanism
  model_ATT.py            # Model definition for the attention mechanism
  train_ATT.py            # Training script for the attention mechanism
  loss_GAN.py             # Loss function for GAN
  model_GAN.py            # Model definition for GAN
  train_GAN.py            # Training script for GAN

Research2/                # Research Direction 2: Identity Recognition and Intruder Detection
  identify/               # Identity recognition model training results
  intruder/               # Intruder detection model training results
  dataloader_identify.py  # Data loader for identity recognition
  model_identify.py       # Model definition for identity recognition
  loss_identify.py        # Loss function for identity recognition
  train_identify.py       # Training script for identity recognition
  plot_identity.py        # Visualization of identity recognition results
  dataloader_intruder.py  # Data loader for intruder detection
  model_intruder.py       # Model definition for intruder detection
  train_intruder.py       # Training script for intruder detection
  plot_intruder.py        # Visualization of intruder detection results

pre_process/              # Data preprocessing module
  SourceData.py           # Source domain data loading
  TargetData.py           # Target domain data loading
  data.py                 # Data processing and grouping
  preprocess.py           # CSI data preprocessing and augmentation
  CSI_fig.py              # CSI data visualization

picture/                  # Project-related figures
```

## Functional Module Description

### Research1 - Attention Mechanism and GAN
- **Attention Mechanism**: Implements cross-domain feature alignment and classification, including components such as feature extractor, multi-head attention module, and cross-attention module.
- **GAN**: Used for generating samples in the target domain, including modules such as feature extractor, generator, and discriminator.

### Research2 - Identity Recognition and Intruder Detection
- **Identity Recognition**: Performs user identity recognition based on CSI data, including components such as feature extraction, attention module, projection head, and classifier.
- **Intruder Detection**: Detects unauthorized user intrusion, including modules such as traditional OpenMax method and learnable threshold detector.

### pre_process - Data Preprocessing
- Performs CSI data loading, normalization, denoising, segmentation, and data augmentation.
- Supports joint processing of source and target domain data.

## Usage Instructions

### Environment Dependencies
- Python 3.8+
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- TorchVision (if image processing is required)

### Install Dependencies
```bash
pip install torch numpy scikit-learn matplotlib torchvision
```

### Train Models
#### Attention Mechanism
```bash
cd Research1
python train_ATT.py
```

#### GAN
```bash
cd Research1
python train_GAN.py
```

#### Identity Recognition
```bash
cd Research2
python train_identify.py
```

#### Intruder Detection
```bash
cd Research2
python train_intruder.py
```

### Visualize Results
#### Identity Recognition Results
```bash
cd Research2
python plot_identity.py
```

#### Intruder Detection Results
```bash
cd Research2
python plot_intruder.py
```

## Model Saving and Loading
During training, models are automatically saved as `.pth` files, with the following paths:
- `Research1/Attention/best_attention_model.pth`
- `Research1/GAN/best_gan_model.pth`
- `Research2/identify/best_identify_model.pth`
- `Research2/intruder/best_intruder_detector.pth`

Models can be loaded using `torch.load()`:
```python
import torch
model = torch.load('Research1/Attention/best_attention_model.pth')
```

## Data Format Description
The CSI data format is `[batch_size, 56, 3, 6000]`, representing:
- `batch_size`: Batch size
- `56`: Number of subcarriers
- `3`: Antenna pair combinations (Tx-Rx)
- `6000`: Time steps

## Notes
- All model training scripts default to using GPU (if available); can be switched to CPU by modifying the `device` parameter.
- Data paths need to be adjusted according to the actual data storage location.
- Visualization scripts depend on the training result file `training_history.json`; ensure to run them after training is completed.

## License
This project is licensed under the MIT License. Please refer to the [LICENSE](LICENSE) file for details.