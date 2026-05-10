"""
PPE Compliance Detection Training Script
======================================

Dataset Overview:
- Source: Roboflow "dd" object-detection dataset v2 (Feb 24, 2024)
- Total Images: 3,521
- Split:
  - TRAIN: 2,567 images (73%)
  - VAL: 614 images (17%)
  - TEST: 340 images (10%)
- Classes:
  - mask
  - glove
  - hairnet
  - maskoff
  - no_glove
  - no_hairnet

Training Configuration:
- Model: YOLOv8n (nano variant for speed)
- Image Size: 640x640
- Batch Size: 16
- Epochs: 100
- Early Stopping: Enabled (patience=20)
- Optimizer: AdamW
- Learning Rate: 0.001
- Weight Decay: 0.0005
- Augmentation: Enabled (Mosaic, MixUp, etc.)
"""

import os
from ultralytics import YOLO
import yaml

def create_dataset_yaml():
    """Create dataset.yaml file for YOLOv8 training"""
    data = {
        'path': 'dataset',  # root dataset directory
        'train': 'train/images',  # train images
        'val': 'valid/images',  # val images
        'test': 'test/images',  # test images
        'names': {
            0: 'mask',
            1: 'glove',
            2: 'hairnet',
            3: 'maskoff',
            4: 'no_glove',
            5: 'no_hairnet'
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def main():
    # Create dataset configuration
    create_dataset_yaml()
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,  # Early stopping patience
        device='0',  # Use GPU if available
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        save=True,
        save_period=10,
        project='runs/train',
        name='ppe_detection',
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    # Export to ONNX
    model.export(format='onnx')

if __name__ == '__main__':
    main() 