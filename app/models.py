from torchvision.models import densenet201
import torch


def get_custom_densenet201(n_classes: int = 16) -> densenet201:
    model = densenet201()
    model.classifier = torch.nn.Linear(in_features=1920, out_features=n_classes, bias=True)
    return model
