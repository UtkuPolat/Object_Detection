from torchvision.datasets import VOCDetection


def load_pascal_voc(root="./data"):
    dataset = VOCDetection(
        root=root,
        year="2012",
        image_set="val",
        download=False,
        transform=None   # to keep PIL images
    )

    return dataset