# Model compilation utilities.

from tensorflow.keras.optimizers import Adam



def compile_model(
        model,
        learning_rate=0.001
):


    model.compile(

        optimizer=Adam(

            learning_rate=learning_rate

        ),

        loss=
        "sparse_categorical_crossentropy",

        metrics=[

            "accuracy"

        ]

    )


    return model
