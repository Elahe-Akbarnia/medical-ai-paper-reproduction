# Runs adversarial robustness experiments.

from utils.metrics import calculate_metrics



class AttackRunner:


    def __init__(
            self,
            model,
            device
    ):

        self.model=model

        self.device=device



    def evaluate_attack(
            self,
            dataloader,
            attack=None
    ):


        targets=[]

        predictions=[]

        probabilities=[]



        for batch in dataloader:


            images=batch["image"].to(
                self.device
            )


            labels=batch["label"].to(
                self.device
            )



            if attack:


                images=attack.generate(

                    self.model,

                    images,

                    labels

                )



            output=self.model(
                images
            )


            logits=output["logits"]



            probs=logits.sigmoid()



            preds=(probs>0.5).int()



            targets.extend(
                labels.cpu().numpy()
            )


            predictions.extend(
                preds.cpu().numpy()
            )


            probabilities.extend(
                probs.detach()
                .cpu()
                .numpy()
            )



        return calculate_metrics(

            targets,

            predictions,

            probabilities

        )
