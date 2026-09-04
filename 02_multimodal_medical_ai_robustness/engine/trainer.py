# Training engine.

# Supports: image model, text model and fusion models.

# Features: checkpoint saving, validation and metric tracking.

import torch

from tqdm import tqdm

from utils.metrics import calculate_metrics

from utils.checkpoint import save_checkpoint



class Trainer:


    def __init__(
            self,
            model,
            optimizer,
            scheduler,
            criterion,
            device,
            logger=None
    ):


        self.model=model

        self.optimizer=optimizer

        self.scheduler=scheduler

        self.criterion=criterion

        self.device=device

        self.logger=logger



    def train_epoch(
            self,
            dataloader
    ):


        self.model.train()


        total_loss=0


        targets=[]

        predictions=[]

        probabilities=[]



        for batch in tqdm(
            dataloader,
            desc="Training"
        ):


            self.optimizer.zero_grad()



            outputs=self.forward_batch(
                batch
            )


            logits=outputs["logits"]



            loss=self.criterion(

                logits.squeeze(),

                batch["label"].to(
                    self.device
                )

            )



            loss.backward()


            self.optimizer.step()



            total_loss += loss.item()



            probs=torch.sigmoid(
                logits
            )

            preds=(probs>0.5).int()



            targets.extend(
                batch["label"].cpu().numpy()
            )


            predictions.extend(
                preds.cpu().numpy()
            )


            probabilities.extend(
                probs.detach().cpu().numpy()
            )



        metrics=calculate_metrics(

            targets,

            predictions,

            probabilities

        )


        metrics["loss"]=(
            total_loss /
            len(dataloader)
        )


        return metrics



    def validate(
            self,
            dataloader
    ):


        self.model.eval()


        targets=[]

        predictions=[]

        probabilities=[]


        total_loss=0



        with torch.no_grad():


            for batch in dataloader:


                outputs=self.forward_batch(
                    batch
                )


                logits=outputs["logits"]



                loss=self.criterion(

                    logits.squeeze(),

                    batch["label"].to(
                        self.device
                    )

                )

                total_loss+=loss.item()



                probs=torch.sigmoid(
                    logits
                )


                preds=(probs>0.5).int()



                targets.extend(
                    batch["label"].cpu().numpy()
                )

                predictions.extend(
                    preds.cpu().numpy()
                )


                probabilities.extend(
                    probs.cpu().numpy()
                )



        metrics=calculate_metrics(

            targets,

            predictions,

            probabilities

        )


        metrics["loss"]=(
            total_loss /
            len(dataloader)
        )


        return metrics



    def fit(
            self,
            train_loader,
            val_loader,
            epochs,
            checkpoint_dir
    ):


        best_f1=0



        for epoch in range(epochs):


            train_metrics=self.train_epoch(
                train_loader
            )


            val_metrics=self.validate(
                val_loader
            )



            if self.scheduler:

                self.scheduler.step(
                    val_metrics["loss"]
                )



            if self.logger:

                self.logger.info(
                    f"""
Epoch {epoch}

Train:
{train_metrics}

Validation:
{val_metrics}
"""
                )



            if val_metrics["f1"] > best_f1:


                best_f1=val_metrics["f1"]



                save_checkpoint(

                    {

                    "epoch":epoch,

                    "model_state_dict":
                    self.model.state_dict(),

                    "optimizer_state_dict":
                    self.optimizer.state_dict(),

                    "scheduler_state_dict":
                    self.scheduler.state_dict()
                    if self.scheduler
                    else None,

                    "best_metric":
                    best_f1

                    },

                    checkpoint_dir,

                    "best_model.pt"

                )



    def forward_batch(
            self,
            batch
    ):

        """
        Handles different model inputs.
        """


        if "frontal" in batch:


            return self.model(

                batch["frontal"].to(
                    self.device
                ),

                batch["lateral"].to(
                    self.device
                ),

            )



        elif "input_ids" in batch:


            return self.model(

                batch["input_ids"].to(
                    self.device
                ),

                batch["attention_mask"].to(
                    self.device
                )

            )
