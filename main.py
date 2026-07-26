from datasets.pascal_voc import load_pascal_voc

from models.yolo import YOLODetector
from models.faster_rcnn import FasterRCNNDetector
from models.detr import DETRDetector

from utils.gt_visualization import show_ground_truth
from utils.yolo_visualization import show_yolo_predictions
from utils.faster_rcnn_visualization import show_faster_predictions
from utils.detr_visualization import show_detr_predictions
from utils.visualization_all import plot_all_predictions


dataset = load_pascal_voc()

image, annotation = dataset[15]

print("Ground Truth")
show_ground_truth(image, annotation)

print("YOLOv11")
yolo = YOLODetector()

yolo_result = yolo.predict(image)

show_yolo_predictions(image, yolo_result)

print("Faster R-CNN")
faster = FasterRCNNDetector()

faster_result = faster.predict(image)

show_faster_predictions(image, faster_result)


print("DETR")
detr = DETRDetector()

detr_result = detr.predict(image)

categories = detr.model.config.id2label

show_detr_predictions(
    image,
    detr_result,
    categories
)

plot_all_predictions(image, annotation, yolo_result, faster_result, detr_result)