import torch
from torch import nn
from torchvision import models

class model(nn.Module):
    def __init__(self, model_name = "resnet50", weights = "DEFAULT", output_shape = [68, 2]):
        super().__init__()
        self.backbone = models.get_model(name = model_name, weights = weights)
        num_filters = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features=num_filters, out_features=output_shape[0]*output_shape[1])
        """
        for param in self.backbone.parameters():
            param.requires_grad=False

        for param in self.backbone.fc.parameters():
            param.requires_grad=True
        
        for param in self.backbone.layer4[2].parameters():
            param.requires_grad=True
        """
    def forward(self, x):
        out = self.backbone(x)
        return out

if __name__ == "__main__":
    input = torch.zeros((64, 3, 224, 224))
    model = model()
    print(model)