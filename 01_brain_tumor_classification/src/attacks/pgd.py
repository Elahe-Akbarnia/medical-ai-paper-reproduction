# Projected Gradient Descent (PGD)

# Iterative adversarial attack


import tensorflow as tf



def pgd_attack(

        model,

        images,

        labels,

        epsilon=0.01,

        alpha=0.002,

        iterations=10

):


    """
    Generate PGD adversarial examples.


    Args:

        epsilon:
            maximum perturbation radius


        alpha:
            step size


        iterations:
            number of attack iterations


    Returns:

        PGD adversarial images

    """



    original_images = tf.identity(

        images

    )


    adversarial_images = tf.identity(

        images

    )



    for _ in range(iterations):


        with tf.GradientTape() as tape:


            tape.watch(

                adversarial_images

            )


            predictions = model(

                adversarial_images,

                training=False

            )


            loss = tf.keras.losses.sparse_categorical_crossentropy(

                labels,

                predictions

            )



        gradient = tape.gradient(

            loss,

            adversarial_images

        )



        signed_gradient = tf.sign(

            gradient

        )



        adversarial_images = (

            adversarial_images

            +

            alpha * signed_gradient

        )



        # Project perturbation back into epsilon ball

        perturbation = (

            adversarial_images

            -

            original_images

        )


        perturbation = tf.clip_by_value(

            perturbation,

            -epsilon,

            epsilon

        )


        adversarial_images = (

            original_images

            +

            perturbation

        )



        # maintain valid image range

        adversarial_images = tf.clip_by_value(

            adversarial_images,

            0.0,

            1.0

        )



    return adversarial_images
