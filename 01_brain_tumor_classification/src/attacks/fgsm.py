#Fast Gradient Sign Method (FGSM)

#Implementation based on:
#Goodfellow et al.
#Explaining and Harnessing Adversarial Examples

#Used in the paper for:
#Brain tumor MRI robustness evaluation



import tensorflow as tf



def fgsm_attack(
        model,
        images,
        labels,
        epsilon=0.01
):

    """
    Generate FGSM adversarial examples.

    Args:

        model:
            trained TensorFlow model

        images:
            input MRI batch

        labels:
            ground truth labels

        epsilon:
            attack strength


    Returns:

        adversarial images
    """


    images = tf.cast(
        images,
        tf.float32
    )


    with tf.GradientTape() as tape:

        tape.watch(images)


        predictions = model(
            images,
            training=False
        )


        loss = tf.keras.losses.sparse_categorical_crossentropy(

            labels,

            predictions

        )



    gradient = tape.gradient(

        loss,

        images

    )



    signed_gradient = tf.sign(

        gradient

    )


    adversarial_images = (

        images

        +

        epsilon * signed_gradient

    )



    # keep valid pixel range

    adversarial_images = tf.clip_by_value(

        adversarial_images,

        0.0,

        1.0

    )


    return adversarial_images
