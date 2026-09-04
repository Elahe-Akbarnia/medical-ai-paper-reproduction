import argparse


from attacks.fgsm import FGSMAttack

from attacks.pgd import PGDAttack



def main():


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--attack",
        choices=[
            "fgsm",
            "pgd"
        ]
    )


    args=parser.parse_args()



    if args.attack=="fgsm":

        attack=FGSMAttack(
            epsilon=8/255
        )


    else:

        attack=PGDAttack(
            epsilon=8/255,
            steps=10
        )


    print(
        "Attack configured:",
        attack
    )


if __name__=="__main__":

    main()
