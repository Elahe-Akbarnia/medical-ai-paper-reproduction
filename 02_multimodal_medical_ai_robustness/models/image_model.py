# SE-ResNet-154 based chest X-ray classifier.

import torch
import torch.nn as nn

import timm



class SEResNet154Classifier(nn.Module):


    def __init__(
            self,
            pretrained=True
    ):

        super().__init__()


        self.backbone = timm.create_model(
            "seresnet154",
            pretrained=pretrained,
            num_classes=0
        )


        feature_dim = self.backbone.num_features


        self.classifier = nn.Linear(
            feature_dim,
            1
        )


    def forward(
            self,
            x
    ):


        features = self.backbone(x)


        logits = self.classifier(
            features
        )


        return {

            "logits": logits,

            "features": features

        }
