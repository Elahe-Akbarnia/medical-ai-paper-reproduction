# Feature squeezing defense

# Implemented techniques:
# 1. Bit-depth reduction
# 2. Gaussian smoothing

# Based on:
# Xu et al.
# Feature Squeezing: Detecting Adversarial Examples


import cv2
import numpy as np
import tensorflow as tf



def reduce_bit_depth(
        images,
        bit_depth=4
):

    """
    Reduce image bit depth.

    Formula:

    X_sq = round(X*(2^b-1))/(2^b-1)

    Args:

        images:
            normalized images [0,1]

        bit_depth:
            number of bits


    Returns:

        squeezed images

    """


    levels = (

        2 ** bit_depth

    ) - 1



    squeezed = tf.round(

        images * levels

    ) / levels



    return squeezed





def gaussian_blur(

        images,

        kernel_size=(3,3),

        sigma=1.0

):

    """
    Apply Gaussian blur.

    Paper:
        kernel size = 3x3
    """



    blurred_images = []



    images_np = images.numpy()



    for image in images_np:


        blurred = cv2.GaussianBlur(

            image,

            kernel_size,

            sigma

        )


        blurred_images.append(

            blurred

        )



    return tf.convert_to_tensor(

        np.array(blurred_images),

        dtype=tf.float32

    )





def feature_squeeze(

        images,

        bit_depth=4,

        kernel_size=(3,3),

        sigma=1.0

):

    """
    Complete feature squeezing pipeline.

    Steps:

    1. Bit reduction
    2. Gaussian smoothing

    """


    squeezed = reduce_bit_depth(

        images,

        bit_depth

    )


    squeezed = gaussian_blur(

        squeezed,

        kernel_size,

        sigma

    )


    return squeezed
