import torch
import torchvision

from torchvision.transforms import ToTensor


class FasterRCNNDetector:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = ToTensor()

    def predict(self, image):

        image = self.transform(image).to(self.device)

        with torch.no_grad():

            prediction = self.model([image])[0]

        return prediction