import argparse

import torch


from utils.checkpoint import load_checkpoint



def main():


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        required=True
    )


    args=parser.parse_args()



    checkpoint=torch.load(
        args.checkpoint
    )


    print(
        "Checkpoint loaded successfully"
    )


if __name__=="__main__":

    main()
