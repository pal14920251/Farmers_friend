import torch
import torch.nn as nn
import torchvision.models as models

# ----------------------------------------------------------
# ResNet50 Model Definition (same architecture as training)
# ----------------------------------------------------------
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(weights=None)   # No pretrained weights
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.network(x)


# ----------------------------------------------------------
# Load Model Function
# ----------------------------------------------------------
def load_model(model_path, num_classes, device="cpu"):
    model = ResNet50Classifier(num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)      # <<--- IMPORTANT FIX
    model.eval()
    return model

