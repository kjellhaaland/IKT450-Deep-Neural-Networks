import timm
from torch import nn
import numpy as np


class FishModel(nn.Module):
    def __init__(self, emb_size=1024):
        super(FishModel, self).__init__()

        # Use resnet50 as the backbone instead of efficientnet
        self.resnet = timm.create_model("resnet50", pretrained=True)

        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=emb_size)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def predict(self, x, refs):
        # Send the image trough the network
        x = self.forward(x)

        # Calculate the euclidean distance between the embeddings of the test image and the reference images
        distances = [np.linalg.norm(x.detach().numpy() - e) for e in refs]

        # Get the index of the reference image with the smallest distance
        pred = np.argmin(distances)

        return pred
