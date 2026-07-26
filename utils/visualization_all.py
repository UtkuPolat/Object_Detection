import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def plot_all_predictions(image, annotation, yolo_res, faster_res, detr_res, object_classes=None, conf_threshold=0.3):
    """
    Plots Ground Truth, YOLOv11, Faster R-CNN, and DETR predictions side by side,
    using a full COCO label dictionary to prevent out-of-range class index errors.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    
    # Standard COCO 80-class mapping used by Torchvision and Hugging Face pre-trained models
    COCO_ID_MAP = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
        10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 
        18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
        38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
        46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
        55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
        64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
        76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
        85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }

    def get_class_name(label_val):
        if isinstance(label_val, str):
            return label_val.lower().strip()
        
        try:
            idx = int(label_val)
        except (ValueError, TypeError):
            return "unknown"

        # 1. Check if it matches COCO standard mapping
        if idx in COCO_ID_MAP:
            return COCO_ID_MAP[idx]

        # 2. Fallback to project-specific object_classes if provided and in range
        if object_classes and 0 <= idx < len(object_classes):
            return str(object_classes[idx])
        
        return f"class_{idx}"

    def draw_boxes(ax, data, title, color="lime", is_ground_truth=False):
        ax.imshow(image)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        
        items_to_draw = []
        
        # 1. Parse Ground Truth
        if is_ground_truth:
            if isinstance(annotation, dict) and "boxes" in annotation:
                for box, cls in zip(annotation.get("boxes", []), annotation.get("labels", [])):
                    items_to_draw.append({"class": str(cls), "bbox": [float(c) for c in box]})
            elif isinstance(annotation, dict) and "annotation" in annotation:
                objects = annotation["annotation"].get("object", [])
                if isinstance(objects, dict):
                    objects = [objects]
                for obj in objects:
                    bnd = obj["bndbox"]
                    items_to_draw.append({
                        "class": str(obj["name"]),
                        "bbox": [float(bnd["xmin"]), float(bnd["ymin"]), float(bnd["xmax"]), float(bnd["ymax"])]
                    })
            elif isinstance(annotation, list):
                items_to_draw = annotation
        
        # 2. Parse YOLOv11
        elif title == "YOLOv11":
            if hasattr(data, "boxes"):
                for box in data.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    if conf >= conf_threshold:
                        cls_id = int(box.cls[0])
                        cls_name = data.names[cls_id] if hasattr(data, "names") else get_class_name(cls_id)
                        items_to_draw.append({"class": cls_name, "bbox": [x1, y1, x2, y2], "confidence": conf})
            elif isinstance(data, list):
                items_to_draw = data

        # 3. Parse Faster R-CNN & 4. Parse DETR
        elif title in ["Faster R-CNN", "DETR"]:
            if isinstance(data, dict) and "boxes" in data:
                boxes = data.get("boxes", [])
                scores = data.get("scores", [])
                labels = data.get("labels", [])
                if torch.is_tensor(boxes): boxes = boxes.detach().cpu().numpy()
                if torch.is_tensor(scores): scores = scores.detach().cpu().numpy()
                if torch.is_tensor(labels): labels = labels.detach().cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if float(score) >= conf_threshold:
                        class_str = get_class_name(label)
                        items_to_draw.append({
                            "class": class_str,
                            "bbox": [float(c) for c in box],
                            "confidence": float(score)
                        })
            elif isinstance(data, list):
                items_to_draw = data

        # Render items onto plot
        for item in items_to_draw:
            if isinstance(item, dict):
                box = item.get("bbox", item.get("box", []))
                label = item.get("class", item.get("label", "unknown"))
                conf = item.get("confidence", item.get("score", None))
                
                if box and len(box) == 4:
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor=color, facecolor="none"
                    )
                    ax.add_patch(rect)
                    
                    text_label = f"{label}" if conf is None else f"{label} {conf:.2f}"
                    ax.text(
                        x1, max(y1 - 5, 10), text_label,
                        color="white", fontsize=9,
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1)
                    )

    # Execute panel drawing
    draw_boxes(axes[0], annotation, "Ground Truth", color="dodgerblue", is_ground_truth=True)
    draw_boxes(axes[1], yolo_res, "YOLOv11", color="lime", is_ground_truth=False)
    draw_boxes(axes[2], faster_res, "Faster R-CNN", color="orange", is_ground_truth=False)
    draw_boxes(axes[3], detr_res, "DETR", color="magenta", is_ground_truth=False)

    plt.tight_layout()
    plt.show()