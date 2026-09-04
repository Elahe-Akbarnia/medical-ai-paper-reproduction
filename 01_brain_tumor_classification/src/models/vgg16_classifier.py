# VGG16-based brain tumor classifier.


from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import (

    GlobalAveragePooling2D,
    Dense,
    Dropout

)

from tensorflow.keras.models import Model



def build_vgg16_classifier(

        input_shape=(128,128,3),

        num_classes=4,

        dropout_rate=0.5

):


    """
    Create VGG16 transfer learning model.
    """


    base_model = VGG16(

        weights="imagenet",

        include_top=False,

        input_shape=input_shape

    )



    # Freeze backbone initially

    base_model.trainable = False



    x = base_model.output


    x = GlobalAveragePooling2D()(x)


    x = Dense(

        128,

        activation="relu"

    )(x)



    x = Dropout(

        dropout_rate

    )(x)



    output = Dense(

        num_classes,

        activation="softmax"

    )(x)



    model = Model(

        inputs=base_model.input,

        outputs=output

    )
    return model
