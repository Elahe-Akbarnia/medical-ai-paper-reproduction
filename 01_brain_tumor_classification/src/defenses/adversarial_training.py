# Adversarial training utilities.

# Combines: clean samples, FGSM samples and PGD samples with feature squeezing.


import tensorflow as tf
import numpy as np


from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

from .feature_squeezing import feature_squeeze




def generate_defense_dataset(

        model,

        X,

        y,

        fgsm_epsilon=0.05,

        pgd_epsilon=0.05,

        pgd_alpha=0.04,

        pgd_iterations=10,

        bit_depth=4,

        kernel_size=(3,3),

        sigma=1.0

):


    """
    Generate adversarial training dataset.


    Returns:

        X_combined
        y_combined

    """


    print(
        "Applying feature squeezing..."
    )


    X_squeezed = feature_squeeze(

        tf.convert_to_tensor(X),

        bit_depth,

        kernel_size,

        sigma

    )



    print(
        "Generating FGSM examples..."
    )


    X_fgsm = fgsm_attack(

        model,

        X_squeezed,

        y,

        epsilon=fgsm_epsilon

    )



    print(
        "Generating PGD examples..."
    )


    X_pgd = pgd_attack(

        model,

        X_squeezed,

        y,

        epsilon=pgd_epsilon,

        alpha=pgd_alpha,

        iterations=pgd_iterations

    )



    X_fgsm = X_fgsm.numpy()

    X_pgd = X_pgd.numpy()



    X_squeezed = X_squeezed.numpy()



    # Combine all samples

    X_combined = np.concatenate(

        [

            X_squeezed,

            X_fgsm,

            X_pgd

        ],

        axis=0

    )



    y_combined = np.concatenate(

        [

            y,

            y,

            y

        ],

        axis=0

    )



    return (

        X_combined,

        y_combined

    )
