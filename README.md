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

## Model Architecture

![EfficientU-Net Architecture](https://github.com/MohsinFurkh/EfficientU-Net/raw/master/images/EfficientU-Net.jpg)

The proposed EfficientU-Net architecture consists of three main components:

1. **Encoder (EfficientNetB7)**:
   - Pre-trained on ImageNet
   - Extracts hierarchical features at different scales
   - Captures both low-level and high-level features

2. **Atrous Convolution (AC) Block**:
   - Captures multi-scale contextual information
   - Uses parallel dilated convolutions with different rates
   - Preserves spatial resolution while expanding the receptive field

3. **Decoder**:
   - Progressive upsampling with skip connections
   - Recovers spatial information from encoder features
   - Produces the final segmentation mask

## Results

### Quantitative Results

| Model             | Dice Score | IoU   | Sensitivity | Specificity |
|-------------------|------------|-------|-------------|-------------|
| Proposed Model    | 0.891      | 0.815 | 0.876       | 0.993       |
| Baseline U-Net    | 0.842      | 0.762 | 0.831       | 0.987       |
| EfficientNetB7    | 0.865      | 0.792 | 0.853       | 0.991       |


### Qualitative Results

![Segmentation Examples](https://github.com/MohsinFurkh/EfficientU-Net/raw/master/images/Fig-12.png)
*Figure 1: Sample segmentation results showing input image, ground truth, and model prediction.*

### Training Curves

![Training Curves](https://github.com/MohsinFurkh/EfficientU-Net/raw/master/images/Fig-13(a).png)
*Figure 2: Training and validation metrics over 50 epochs.*

### Ablation Study

| Model Variant       | Dice Score | Parameters (M) |
|---------------------|------------|----------------|
| Base U-Net          | 0.842      | 31.0           |
| + EfficientNetB7    | 0.865      | 64.5           |
| + AC Block          | 0.891      | 67.2           |

*Table 2: Ablation study showing the impact of each component.*

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

## Dataset

The model was trained on the [Dataset Name] dataset. For access to the dataset, please contact [Contact Information].

## Citation

If you use this implementation in your research, please cite:

```
@article{efficientunet2023,
  title={EfficientU-Net: A Novel Deep Learning Method for Breast Tumor Segmentation},
  author={[Your Names]},
  journal={Neural Processing Letters},
  year={2023},
  doi={10.1007/s11063-023-11333-x}
}
```
