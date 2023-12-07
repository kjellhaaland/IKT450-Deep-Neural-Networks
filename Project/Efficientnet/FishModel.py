import timm
from torch import nn
import numpy as np


class FishModel(nn.Module):
    def __init__(self, out_features=512):
        super(FishModel, self).__init__()

        self.efficientnet = timm.create_model("efficientnet_b0", pretrained=True)

        input_features = self.efficientnet.classifier.in_features

        self.efficientnet.classifier = nn.Linear(in_features=input_features, out_features=out_features)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

    def predict(self, x, refs):
        # Send the image trough the network
        x = self.forward(x)

        # Calculate the euclidean distance between the embeddings of the test image and the reference images
        distances = [np.linalg.norm(x.detach().numpy() - e) for e in refs]

        # Get the index of the reference image with the smallest distance
        pred = np.argmin(distances)

        return pred
