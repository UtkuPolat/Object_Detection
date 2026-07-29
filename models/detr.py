import torch

from transformers import DetrImageProcessor
from transformers import DetrForObjectDetection


class DETRDetector:

    def __init__(self, confidence_threshold=0.5):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )

        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50"
        )

        self.model.to(self.device)
        self.model.eval()

        self.confidence_threshold = confidence_threshold

    def predict(self, image):

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [image.size[::-1]],   # (height, width)
            device=self.device
        )

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        return results