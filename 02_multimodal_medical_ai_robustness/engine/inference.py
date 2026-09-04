import torch



class Predictor:


    def __init__(
            self,
            model,
            device
    ):

        self.model=model

        self.device=device



    def predict(
            self,
            batch
    ):


        self.model.eval()


        with torch.no_grad():


            output=self.model(
                **batch
            )


            probability=torch.sigmoid(
                output["logits"]
            )


        return probability.item()
