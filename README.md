# Object Detection Model Comparison

A comparative study of three state-of-the-art object detection models:

- YOLOv11
- Faster R-CNN
- DETR (Detection Transformer)

The project evaluates these models on the Pascal VOC dataset by comparing their detection performance, inference speed, and qualitative predictions.

---

## Features

- Ground truth visualization
- YOLOv11 inference
- Faster R-CNN inference
- DETR inference
- Prediction visualization

---

## Project Structure

```
Object_Detection/
│
├── datasets/
│   ├── pascal_voc.py
│   └── ...
│
├── models/
│   ├── yolo.py
│   ├── faster_rcnn.py
│   └── detr.py
│
├── utils/
│   ├── gt_visualization.py
│   ├── yolo_visualization.py
│   ├── faster_rcnn_visualization.py
│   └── detr_visualization.py
│
├── weights/
│
├── main.py
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Models

### YOLOv11

- One-stage detector
- Real-time object detection
- Fast inference
- Ultralytics implementation

### Faster R-CNN

- Two-stage detector
- High localization accuracy
- Region Proposal Network (RPN)
- torchvision implementation

### DETR

- Transformer-based detector
- End-to-end object detection
- Eliminates anchor boxes and Non-Maximum Suppression (NMS)
- Hugging Face Transformers implementation

---

## Dataset

The project uses the **Pascal VOC 2012** dataset.

Download:

https://host.robots.ox.ac.uk/pascal/VOC/

Extract the dataset and place it inside the dataset directory.

Example:

```
datasets/
└── VOCdevkit/
    └── VOC2012/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/UtkuPolat/Object_Detection.git
cd Object_Detection
```

Create the conda environment

```bash
conda env create -f environment.yml
```

Activate it

```bash
conda activate source
```

---

## Download Model Weights

The first execution automatically downloads the pretrained weights if they are not already available.

Alternatively, download them manually:

- YOLOv11
- Faster R-CNN
- DETR

and place them inside the `weights/` directory.

---

## Running

Run the project using

```bash
python main.py
```

The program will

- Load Pascal VOC images
- Display ground truth
- Run YOLOv11
- Run Faster R-CNN
- Run DETR
- Visualize predictions


## Author

**Utku Polat**

GitHub: https://github.com/UtkuPolat