# Utilities for VGG16 fine tuning


def unfreeze_last_layers(

        model,

        number_of_layers=10

):

    """
    Unfreeze last N VGG16 layers.
    """


    for layer in model.layers:

        layer.trainable = False



    for layer in model.layers[-number_of_layers:]:

        layer.trainable = True



    return model
