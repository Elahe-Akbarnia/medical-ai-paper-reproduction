# Train adversarially defended model.
# Uses: FGSM, PGD and Feature Squeezing.


import os
import tensorflow as tf



from src.data.dataset_loader import (
    load_dataset
)


from src.defenses.adversarial_training import (

    generate_defense_dataset

)



from src.models.compile_model import (

    compile_model

)



MODEL_INPUT = (

    "checkpoints/baseline_model.keras"

)



OUTPUT_MODEL = (

    "checkpoints/robust_model.keras"

)



DATASET_PATH = (

    "data/raw"

)



def main():


    print(

        "Loading baseline model..."

    )



    model = tf.keras.models.load_model(

        MODEL_INPUT

    )



    print(

        "Loading dataset..."

    )



    X,y = load_dataset(

        DATASET_PATH

    )



    print(

        "Generating adversarial training data..."

    )



    X_adv, y_adv = generate_defense_dataset(

        model,

        X,

        y,


        # Simulation 1

        fgsm_epsilon=0.05,


        pgd_epsilon=0.05,


        pgd_alpha=0.04,


        pgd_iterations=10,


        bit_depth=4,


        kernel_size=(3,3),


        sigma=1.0

    )



    print(

        "Defense dataset:",

        X_adv.shape

    )



    compile_model(

        model

    )



    print(

        "Adversarial training..."

    )



    model.fit(

        X_adv,

        y_adv,

        epochs=10,

        batch_size=32,

        validation_split=0.2

    )



    os.makedirs(

        "checkpoints",

        exist_ok=True

    )



    model.save(

        OUTPUT_MODEL

    )


    print(

        "Saved robust model"

    )



if __name__ == "__main__":

    main()
