# Generate FGSM and PGD adversarial datasets


import numpy as np
import tensorflow as tf


from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack



def generate_attacks(

        model,

        X_test,

        y_test

):


    print(
        "Generating FGSM examples..."
    )


    fgsm_examples = fgsm_attack(

        model,

        X_test,

        y_test,

        epsilon=0.01

    )



    print(
        "Generating PGD examples..."
    )


    pgd_examples = pgd_attack(

        model,

        X_test,

        y_test,

        epsilon=0.01,

        alpha=0.002,

        iterations=10

    )


    return (

        np.array(fgsm_examples),

        np.array(pgd_examples)

    )
