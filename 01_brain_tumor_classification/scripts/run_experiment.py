# Unified experiment launcher.


import argparse

import subprocess




def main():


    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--mode",

        required=True,

        choices=[

            "baseline",

            "defense",

            "evaluate"

        ]

    )


    args = parser.parse_args()



    if args.mode=="baseline":


        subprocess.run(

            [

                "python",

                "scripts/train_baseline.py"

            ]

        )



    elif args.mode=="defense":


        subprocess.run(

            [

                "python",

                "scripts/train_defense.py"

            ]

        )



    elif args.mode=="evaluate":


        subprocess.run(

            [

                "python",

                "scripts/evaluate.py"

            ]

        )




if __name__=="__main__":

    main()
