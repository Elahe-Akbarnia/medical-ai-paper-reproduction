# BioClinicalBERT classifier.



import torch.nn as nn

from transformers import AutoModel



class BioClinicalBERTClassifier(nn.Module):


    def __init__(
            self,
            pretrained_name=
            "emilyalsentzer/Bio_ClinicalBERT"
    ):

        super().__init__()


        self.encoder = AutoModel.from_pretrained(
            pretrained_name
        )


        hidden_dim = (
            self.encoder.config.hidden_size
        )


        self.classifier = nn.Linear(
            hidden_dim,
            1
        )


    def forward(
            self,
            input_ids,
            attention_mask
    ):


        outputs = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )


        cls_embedding = (
            outputs.last_hidden_state[:,0,:]
        )


        logits = self.classifier(
            cls_embedding
        )


        return {

            "logits": logits,

            "features": cls_embedding

        }
