from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def predict(self, image):
        # Run inference
        results = self.model(image, verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # Extract [xmin, ymin, xmax, ymax]
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                label_name = self.model.names[cls_id]
                detections.append({
                    "box": [float(c) for c in box],
                    "label": str(label_name),
                    "confidence": float(conf),
                })

        return detections

