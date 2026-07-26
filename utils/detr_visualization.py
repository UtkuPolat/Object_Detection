import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_detr_predictions(image, prediction, categories):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image)

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]

    for box, label, score in zip(boxes, labels, scores):

        x1, y1, x2, y2 = box.tolist()

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            f"{categories[int(label)]} {score:.2f}",
            color="cyan",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.7)
        )

    plt.axis("off")
    plt.show()