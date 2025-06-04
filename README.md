# EfficientU-Net for Breast Tumor Segmentation

This repository contains the implementation of EfficientU-Net, a novel deep learning method for breast tumor segmentation in ultrasound (US) images. The model combines the power of EfficientNetB7 as an encoder with an Atrous Convolution (AC) block to improve segmentation accuracy.

## Architecture

The EfficientU-Net architecture consists of:
- **Encoder**: EfficientNetB7 pre-trained on ImageNet
- **AC Block**: Atrous Convolution block with multiple dilation rates
- **Decoder**: Standard U-Net decoder with skip connections

## Prerequisites

- Python 3.7+
- TensorFlow 2.8.0+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MohsinFurkh/EfficientU-Net.git
cd EfficientU-Net
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Prepare your dataset in the appropriate format (images and corresponding masks).
2. Update the `load_data()` function in `train.py` to load your dataset.
3. Run the training script:
```bash
python train.py
```

### Model Architecture

To view the model architecture:
```python
from model import build_efficientunet
model = build_efficientunet()
model.summary()
```

## Results

[Add your training/validation metrics and segmentation results here]

## Citation

If you use this code in your research, please cite:

```
@article{efficientunet2023,
  title={EfficientU-Net: A Novel Deep Learning Method for Breast Tumor Segmentation},
  author={[Your Names]},
  journal={Neural Processing Letters},
  year={2023},
  doi={10.1007/s11063-023-11333-x}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
