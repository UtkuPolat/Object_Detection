import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision


def show_faster_predictions(image, prediction, threshold=0.5):

    fig, ax = plt.subplots(figsize=(8,8))

    ax.imshow(image)

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    for box, score, label in zip(boxes, scores, labels):

        if score < threshold:
            continue

        x1, y1, x2, y2 = box.tolist()

        rect = patches.Rectangle(
            (x1, y1),
            x2-x1,
            y2-y1,
            linewidth=2,
            edgecolor="blue",
            facecolor="none"
        )

        ax.add_patch(rect)
        
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        categories = weights.meta["categories"]
        label_name = categories[int(label)]

        ax.text(
            x1,
            y1-5,
            f"{label_name} {score:.2f}",
            color="blue",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7)
        )

    plt.axis("off")
    plt.show()