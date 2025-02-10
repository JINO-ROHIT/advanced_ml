import torch
import torch.nn as nn
from torchvision import models, transforms
from loguru import logger

class Inference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.__load_labels()
        self.__load_model()
    
    def __repr__(self):
        return "Inference engine ready for predictions"
    
    def __load_model(self):
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model = self.model.to(self.device)

        import os
        weights_path = os.path.join(os.path.dirname(__file__), '../weights/best_model_params.pt')
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict = True)

        logger.info("Model weights loaded")
    
    def __transform_image(self, data):
        data_transforms = {
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return data_transforms['val'](data).unsqueeze(0).to(self.device)
    
    def __load_labels(self):
        import json

        import os
        labels_path = os.path.join(os.path.dirname(__file__), '../weights/label.json')
        with open(labels_path, "r") as file:
            self.labels = json.load(file)
        
        logger.info("labels are loaded")

    def __call__(self, data):
        data = self.__transform_image(data)
        with torch.no_grad():
            output = self.model(data)
            _, predicted = torch.max(output, 1)
        return self.labels[str(predicted.item())]


# if __name__ == '__main__':
#     from PIL import Image
#     test_data = Image.open('../train_data/hymenoptera_data/train/ants/0013035.jpg')
#     engine = Inference()
#     print(engine)
#     prediction = engine(test_data)
#     print(f"Prediction: {prediction}")
