import json
import os

import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

DEFAULT_CLASSES = [
    "Battery",
    "Cardboard",
    "Clothes",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
]


class GarbageClassifier:
    def __init__(
        self,
        model_path="Garbage_Classification_Outputs/best_model.pth",
        classes_path="Garbage_Classification_Outputs/classes.json",
        device="cpu",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.classes = self._load_classes(classes_path)
        
        class TransferLearningModel(torch.nn.Module):
            def __init__(self, num_classes):
                super().__init__()

                self.backbone = models.resnet18(weights=None)

                num_features = self.backbone.fc.in_features

                self.backbone.fc = torch.nn.Sequential(
                    torch.nn.Linear(num_features, 256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(256, num_classes)
                )

            def forward(self, x):
                return self.backbone(x)

        self.model = TransferLearningModel(
            num_classes=len(self.classes)
        )


        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_classes(self, classes_path):
        if classes_path and os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return DEFAULT_CLASSES

    def predict(self, image_path):
        """Predict garbage class from image path."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        return {
            "class": self.classes[prediction.item()],
            "confidence": float(confidence.item()),
            "all_probabilities": {
                self.classes[i]: float(probabilities[0, i].item())
                for i in range(len(self.classes))
            },
        }

    def predict_batch(self, image_paths):
        """Batch prediction for a list of image paths."""
        return [self.predict(path) for path in image_paths]


if __name__ == "__main__":
    classifier = GarbageClassifier("garbage_model.pth")
    result = classifier.predict("test_image.jpg")
    print(result)
