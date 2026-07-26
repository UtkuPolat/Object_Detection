import matplotlib.patches as patches
import matplotlib.pyplot as plt


def show_yolo_predictions(image, result):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image)

    # Handle if result is returned as a list of dicts from your wrapper
    if isinstance(result, list):
        for item in result:
            box = item.get("bbox", item.get("box"))
            label = item.get("class", item.get("label", "unknown"))
            confidence = float(item.get("confidence", item.get("score", 1.0)))

            if box and len(box) == 4:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 5,
                    f"{label} {confidence:.2f}",
                    color="lime",
                    fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.6),
                )
    else:
        # Fallback if it's the raw Ultralytics results object
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf)
            cls = int(box.cls)
            label = result.names[cls]

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                f"{label} {confidence:.2f}",
                color="lime",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.6),
            )

    plt.axis("off")
    plt.show()