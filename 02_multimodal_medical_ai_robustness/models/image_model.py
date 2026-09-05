
import torch
import torch.nn as nn
import timm



class SEResNet152Classifier(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()


        self.backbone = timm.create_model(
            "seresnet152d.ra2_in1k",
            pretrained=pretrained,
            num_classes=0
        )


        self.feature_dim = self.backbone.num_features


        self.classifier = nn.Linear(
            self.feature_dim,
            1
        )


    def forward(self, x):

        features = self.backbone(x)


        logits = self.classifier(
            features
        )


        return {
            "features": features,
            "logits": logits
        }



class DualViewImageModel(nn.Module):

    def __init__(self):

        super().__init__()


        self.frontal_model = SEResNet152Classifier()

        self.lateral_model = SEResNet152Classifier()



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
