# Evaluate clean and adversarial robustness.


import tensorflow as tf

from sklearn.metrics import (

    classification_report,

    accuracy_score

)


from src.data.dataset_loader import (

    load_dataset

)


from src.attacks.fgsm import fgsm_attack


from src.attacks.pgd import pgd_attack




MODEL_PATH = (

    "checkpoints/robust_model.keras"

)


DATASET_PATH = (

    "data/raw"

)




def evaluate(

        model,

        X,

        y,

        name

):


    predictions = model.predict(

        X

    )


    labels = predictions.argmax(

        axis=1

    )



    print(

        "\n",

        name

    )


    print(

        classification_report(

            y,

            labels

        )

    )



def main():


    model = tf.keras.models.load_model(

        MODEL_PATH

    )



    X,y = load_dataset(

        DATASET_PATH

    )



    # Clean

    evaluate(

        model,

        X,

        y,

        "Clean"

    )



    # FGSM

    X_fgsm = fgsm_attack(

        model,

        X,

        y,

        epsilon=0.01

    )



    evaluate(

        model,

        X_fgsm.numpy(),

        y,

        "FGSM"

    )



    # PGD

    X_pgd = pgd_attack(

        model,

        X,

        y,

        epsilon=0.01,

        alpha=0.002,

        iterations=10

    )



    evaluate(

        model,

        X_pgd.numpy(),

        y,

        "PGD"

    )





if __name__ == "__main__":

    main()
