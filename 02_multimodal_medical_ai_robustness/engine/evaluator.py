import torch

from utils.metrics import calculate_metrics



class Evaluator:


    def __init__(
            self,
            model,
            device
    ):

        self.model=model

        self.device=device



    def evaluate(
            self,
            dataloader
    ):


        self.model.eval()


        targets=[]

        predictions=[]

        probabilities=[]



        with torch.no_grad():


            for batch in dataloader:


                output=self.model(
                    **batch
                )


                logits=output["logits"]



                probs=torch.sigmoid(
                    logits
                )


                preds=(
                    probs>0.5
                ).int()



                targets.extend(
                    batch["label"].numpy()
                )


                predictions.extend(
                    preds.cpu().numpy()
                )


                probabilities.extend(
                    probs.cpu().numpy()
                )



        return calculate_metrics(

            targets,

            predictions,

            probabilities

        )
