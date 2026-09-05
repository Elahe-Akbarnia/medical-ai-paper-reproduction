import argparse

import torch

from torch.optim import Adam

from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils.config import Config

from utils.seed import set_seed

from losses.classification_loss import get_loss

from engine.trainer import Trainer

from src.build_model import build_model

from src.build_dataset import (
    prepare_dataframe,
    build_image_loaders
)


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--config",
        required=True
    )


    parser.add_argument(
        "--model",
        required=True
    )


    parser.add_argument(
        "--reports",
        required=True
    )


    parser.add_argument(
        "--projections",
        required=True
    )


    parser.add_argument(
        "--images",
        required=True
    )


    args = parser.parse_args()



    cfg = Config(
        args.config
    )


    set_seed(
        cfg.get(
            "experiment.seed",
            42
        )
    )


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )



    # -----------------------
    # Dataset
    # -----------------------

    dataframe = prepare_dataframe(

        args.reports,

        args.projections,

        args.images

    )


    print(
        "Samples:",
        len(dataframe)
    )



    train_loader, val_loader = build_image_loaders(

        dataframe,

        image_size=cfg.get(
            "data.image_size",
            224
        ),

        batch_size=cfg.get(
            "training.batch_size",
            8
        )

    )



    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )



    # -----------------------
    # Model
    # -----------------------

    model = build_model(
        args.model
    )


    model = model.to(
        device
    )



    # -----------------------
    # Optimization
    # -----------------------

    optimizer = Adam(

        model.parameters(),

        lr=cfg.get(
            "training.learning_rate",
            0.0001
        )

    )


    scheduler = ReduceLROnPlateau(
        optimizer
    )


    criterion = get_loss()



    # -----------------------
    # Training
    # -----------------------

    trainer = Trainer(

        model,

        optimizer,

        scheduler,

        criterion,

        device

    )


    trainer.fit(

        train_loader,

        val_loader,

        cfg.get(
            "training.epochs",
            13
        ),

        "checkpoints"

    )



if __name__ == "__main__":

    main()
