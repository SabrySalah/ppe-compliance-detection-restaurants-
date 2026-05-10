import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
 
class PPEViolationDetector:
    def __init__(self, model_path, output_dir="violations"):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
 
        self.class_names = {
            0: 'glove',
            1: 'hairnet',
            2: 'maskoff',
            3: 'maskon',
            4: 'no_glove',
            5: 'no_hairnet'
        }
 
        # العتبات الخاصة بكل كلاس
        self.conf_thresholds = {
            0: 0.5,  # glove
            1: 0.6,  # hairnet
            2: 0.6,  # maskoff
            3: 0.6,  # maskon
            4: 0.5,  # no_glove
            5: 0.7   # no_hairnet
        }
 
        self.violation_classes = [2, 4, 5]
        self.min_interval = 2.0
        self.last_save = 0
 
        os.makedirs(self.output_dir, exist_ok=True)
 
    def process_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        violations = []
 
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
 
            threshold = self.conf_thresholds.get(cls_id, 0.5)  # عتبة الكلاس أو القيمة الافتراضية
 
            if conf < threshold:
                continue
 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.class_names[cls_id]
 
            print(f"🟢 Detected: {label} (class {cls_id}) with confidence {conf:.2f}")
            color = (0, 0, 255) if cls_id in self.violation_classes else (0, 255, 0)
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
 
            if cls_id in self.violation_classes:
                violations.append(label)
 
        if violations:
            self.save_violation(frame, violations)
 
        return frame
 
    def save_violation(self, frame, labels):
        current_time = time.time()
        if (current_time - self.last_save) >= self.min_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violation_{timestamp}_{'_'.join(labels)}.jpg"
            path = os.path.join(self.output_dir, filename)
            cv2.imwrite(path, frame)
            print(f"⚠️ Violation ({', '.join(labels)}) saved to {path}")
            self.last_save = current_time
 
            message_body = f"""
⚠️ Violation Detected!
🕑 Time: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
            print(message_body)
 
def main():
    model_path = "C:/Users/Lenovo/Desktop/personal projects/kitchen_ppe_optimized/pt/91best.pt"
    video_path = "C:/Users/Lenovo/Desktop/2025-05-21 20-01-45.mkv"
    detector = PPEViolationDetector(model_path)
    cap = cv2.VideoCapture(video_path)
 
    if not cap.isOpened():
        print("❌ Error: Could not open video source.")
        return
 
    print("🚨 PPE Violation Monitoring... Press 'q' to exit.")
    last_processed_time = time.time()
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        current_time = time.time()
        if current_time - last_processed_time >= 1.0:
            annotated = detector.process_frame(frame)
            last_processed_time = current_time
        else:
            annotated = frame
 
        cv2.imshow("PPE Monitor", annotated)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
 