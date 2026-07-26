import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_ground_truth(image, annotation):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image)

    objects = annotation["annotation"]["object"]

    if not isinstance(objects, list):
        objects = [objects]

    for obj in objects:

        box = obj["bndbox"]

        xmin = int(box["xmin"])
        ymin = int(box["ymin"])
        xmax = int(box["xmax"])
        ymax = int(box["ymax"])

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymin - 5,
            obj["name"],
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7)
        )

    plt.axis("off")
    plt.show()