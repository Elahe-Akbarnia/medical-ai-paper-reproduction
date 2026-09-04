# VisionBERT Early Fusion model.

import torch
import torch.nn as nn



class VisionBERTEarlyFusion(nn.Module):


    def __init__(
            self,
            image_model,
            text_model
    ):

        super().__init__()


        self.image_model = image_model

        self.text_model = text_model



        self.classifier = nn.Sequential(

            nn.Linear(
                4864,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                512,
                1
            )

        )



    def forward(
            self,
            frontal,
            lateral,
            input_ids,
            attention_mask
    ):


        image_output = self.image_model(

            frontal,

            lateral

        )


        text_output = self.text_model(

            input_ids,

            attention_mask

        )


        fused_features = torch.cat(

            [

            image_output["features"],

            text_output["features"]

            ],

            dim=1

        )


        logits = self.classifier(
            fused_features
        )


        return {


            "logits": logits,


            "features": fused_features

        }
