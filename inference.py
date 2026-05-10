"""
PPE Compliance Detection Inference Script
=======================================

This script performs real-time PPE compliance detection using a trained YOLOv8 model.
It processes video input (camera or file) and provides:
- Bounding box detection for PPE items
- Person counting
- Violation tracking (missing PPE items)
- Real-time display of statistics

Requirements:
- OpenCV
- Ultralytics YOLOv8
- Numpy
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class PPEComplianceDetector:
    def __init__(self, model_path='runs/train/ppe_detection/weights/best.pt'):
        """Initialize the PPE compliance detector"""
        self.model = YOLO(model_path)
        self.violations = defaultdict(int)
        self.total_persons = 0
        
        # Define PPE requirements
        self.required_ppe = {
            'mask': True,
            'glove': True,
            'hairnet': True
        }
        
        # Define violation classes
        self.violation_classes = {
            'maskoff': 'mask',
            'no_glove': 'glove',
            'no_hairnet': 'hairnet'
        }
        
        # Colors for visualization
        self.colors = {
            'mask': (0, 255, 0),      # Green
            'glove': (0, 255, 0),     # Green
            'hairnet': (0, 255, 0),   # Green
            'maskoff': (0, 0, 255),   # Red
            'no_glove': (0, 0, 255),  # Red
            'no_hairnet': (0, 0, 255) # Red
        }

    def process_frame(self, frame):
        """Process a single frame and return annotated frame with statistics"""
        # Run YOLOv8 inference
        results = self.model(frame)
        
        # Reset counters
        self.violations = defaultdict(int)
        self.total_persons = 0
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:  # Confidence threshold
                    continue
                    
                # Get class name
                class_name = self.model.names[cls]
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = self.colors.get(class_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update statistics
                if class_name in self.violation_classes:
                    self.violations[self.violation_classes[class_name]] += 1
                elif class_name in self.required_ppe:
                    self.total_persons += 1
        
        # Add statistics overlay
        self._add_statistics_overlay(frame)
        
        return frame

    def _add_statistics_overlay(self, frame):
        """Add statistics overlay to the frame"""
        # Calculate total violations
        total_violations = sum(self.violations.values())
        
        # Add statistics text
        stats = [
            f"Total Persons: {self.total_persons}",
            f"Total Violations: {total_violations}",
            f"Missing Mask: {self.violations['mask']}",
            f"Missing Gloves: {self.violations['glove']}",
            f"Missing Hairnet: {self.violations['hairnet']}"
        ]
        
        # Draw statistics
        for i, stat in enumerate(stats):
            y = 30 + i * 30
            cv2.putText(frame, stat, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def main():
    # Initialize detector
    detector = PPEComplianceDetector()
    
    # Initialize video capture (0 for webcam, or path to video file)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Starting PPE compliance detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Display frame
        cv2.imshow('PPE Compliance Detection', processed_frame)
        
        # Print statistics to console
        print(f"\rTotal Persons: {detector.total_persons} | "
              f"Violations: {sum(detector.violations.values())} | "
              f"Missing Mask: {detector.violations['mask']} | "
              f"Missing Gloves: {detector.violations['glove']} | "
              f"Missing Hairnet: {detector.violations['hairnet']}", end='')
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 