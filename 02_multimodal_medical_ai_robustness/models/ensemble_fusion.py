# VisionBERT Ensemble Fusion.

import torch.nn as nn



class VisionBERTEnsembleFusion(nn.Module):


    def __init__(
            self,
            image_model,
            text_model
    ):

        super().__init__()


        self.image_model=image_model

        self.text_model=text_model



    def forward(
            self,
            frontal,
            lateral,
            input_ids,
            attention_mask
    ):


        image_output=self.image_model(
            frontal,
            lateral
        )


        text_output=self.text_model(
            input_ids,
            attention_mask
        )


        logits = (

            image_output["logits"]

            +

            text_output["logits"]

        )


        return {

            "logits": logits

        }
