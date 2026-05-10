# Kitchen PPE Compliance Detection

This project implements a real-time PPE (Personal Protective Equipment) compliance detection system for kitchen environments using YOLOv8. It detects and tracks violations of PPE requirements including masks, gloves, and hairnets.

## Dataset

The model is trained on the Roboflow "dd" object-detection dataset (version 2, generated Feb 24, 2024):
- Total images: 3,521
- Training split: 2,567 images (73%)
- Validation split: 614 images (17%)
- Test split: 340 images (10%)

Classes:
- mask
- glove
- hairnet
- maskoff
- no_glove
- no_hairnet

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset from Roboflow and place it in the following structure:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Usage

### Training

To train the model:
```bash
python train.py
```

The training script will:
- Create the dataset configuration
- Train the YOLOv8 model
- Save the best weights
- Export the model to ONNX format

### Inference

To run real-time PPE compliance detection:
```bash
python inference.py
```

The inference script will:
- Load the trained model
- Process video input (webcam by default)
- Display real-time detections and statistics
- Print violation counts to the console

To use a video file instead of webcam, modify the `cap = cv2.VideoCapture(0)` line in `inference.py` to point to your video file.

## Features

- Real-time PPE compliance detection
- Violation tracking (missing mask, gloves, or hairnet)
- Person counting
- Visual feedback with bounding boxes and labels
- Console output of statistics
- Export to ONNX for deployment

## Performance

The model is configured for optimal performance:
- Image size: 640x640
- Batch size: 16
- Early stopping enabled
- Automatic hyperparameter tuning
- Optimized for real-time inference

## License

This project is licensed under the MIT License - see the LICENSE file for details. 