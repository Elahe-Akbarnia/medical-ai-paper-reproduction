import argparse

import torch

from torch.optim import Adam

from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils.config import Config

from utils.seed import set_seed

from utils.logger import create_logger


from losses.classification_loss import get_loss

from engine.trainer import Trainer


from src.build_model import build_model



def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--config",
        required=True
    )


    parser.add_argument(
        "--model",
        required=True
    )


    args=parser.parse_args()



    cfg=Config(
        args.config
    )



    set_seed(
        cfg.get(
            "experiment.seed",
            42
        )
    )


    device="cuda" if torch.cuda.is_available() else "cpu"



    model=build_model(
        args.model
    )


    model=model.to(device)



    optimizer=Adam(

        model.parameters(),

        lr=cfg.get(
            "training.learning_rate"
        )

    )


    scheduler=ReduceLROnPlateau(
        optimizer
    )



    criterion=get_loss()



    logger=create_logger(

        args.model,

        f"results/logs/{args.model}.log"

    )



    trainer=Trainer(

        model,

        optimizer,

        scheduler,

        criterion,

        device,

        logger

    )


    print(
        "Training pipeline ready"
    )


if __name__=="__main__":

    main()
