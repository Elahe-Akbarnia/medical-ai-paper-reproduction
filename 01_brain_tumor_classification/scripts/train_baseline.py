# Train baseline VGG16 brain tumor classifier.

# Pipeline:
# 1. Load MRI dataset
# 2. Build VGG16
# 3. Train frozen backbone
# 4. Fine tune last layers
# 5. Save model


import os
import tensorflow as tf



from src.data.dataset_loader import (
    load_dataset,
    create_train_validation_split
)


from src.models.vgg16_classifier import (
    build_vgg16_classifier
)


from src.models.compile_model import (
    compile_model
)


from src.models.finetune import (
    unfreeze_last_layers
)


from src.utils.seed import set_seed



# -------------------------
# Configuration
# -------------------------

DATASET_PATH = "data/raw"

MODEL_PATH = "checkpoints/baseline_model.keras"


BATCH_SIZE = 32


SEED = 42



# Main

def main():


    set_seed(SEED)



    print(
        "Loading dataset..."
    )


    X,y = load_dataset(

        DATASET_PATH

    )



    X_train, X_val, y_train, y_val = create_train_validation_split(

        X,

        y,

        validation_size=0.2,

        seed=SEED

    )



    print(

        "Training samples:",

        X_train.shape[0]

    )


    print(

        "Validation samples:",

        X_val.shape[0]

    )



    # Build model

    model = build_vgg16_classifier()



    model = compile_model(

        model

    )



    # Stage 1
    # Frozen VGG16

    print(

        "Stage 1: Frozen backbone training"

    )



    model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=5,

        batch_size=BATCH_SIZE

    )



    # Stage 2
    # Fine tuning

    print(

        "Stage 2: Fine tuning"

    )


    unfreeze_last_layers(

        model,

        number_of_layers=10

    )


    model = compile_model(

        model

    )



    model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=10,

        batch_size=BATCH_SIZE

    )



    os.makedirs(

        "checkpoints",

        exist_ok=True

    )



    model.save(

        MODEL_PATH

    )


    print(

        "Saved:",

        MODEL_PATH

    )



if __name__ == "__main__":

    main()
