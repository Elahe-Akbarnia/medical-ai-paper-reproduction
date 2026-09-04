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

class DualViewImageModel(nn.Module):


    def __init__(self):

        super().__init__()


        self.frontal_model = SEResNet154Classifier()


        self.lateral_model = SEResNet154Classifier()



    def forward(
            self,
            frontal,
            lateral
    ):


        frontal_output = self.frontal_model(
            frontal
        )


        lateral_output = self.lateral_model(
            lateral
        )


        features = torch.cat(
            [
                frontal_output["features"],
                lateral_output["features"]
            ],
            dim=1
        )


        return {


            "features": features,


            "frontal_logits":
                frontal_output["logits"],


            "lateral_logits":
                lateral_output["logits"]

        }
