# VisionBERT Late Fusion.

import torch.nn as nn



class VisionBERTLateFusion(nn.Module):


    def __init__(
            self,
            image_model,
            text_model
    ):

        super().__init__()


        self.image_model=image_model

        self.text_model=text_model



        self.fusion_classifier = nn.Sequential(

            nn.Linear(
                4,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                2
            )

        )



    def forward(
            self,
            frontal,
            lateral,
            input_ids,
            attention_mask
    ):


        image_out=self.image_model(
            frontal,
            lateral
        )


        text_out=self.text_model(
            input_ids,
            attention_mask
        )


        image_logits=image_out["logits"]

        text_logits=text_out["logits"]


        combined = torch.cat(

            [

            image_logits,

            text_logits

            ],

            dim=1

        )


        output=self.fusion_classifier(
            combined
        )


        return {

            "logits":output

        }
