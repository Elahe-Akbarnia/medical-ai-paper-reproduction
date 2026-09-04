# MRI image preprocessing utilities.
# The paper resizes MRI images to 128x128 and normalizes pixel values to [0,1].


import cv2
import numpy as np



def resize_image(image, size=(128,128)):

    """
    Resize MRI image.
    """

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )



def normalize_image(image):

    """
    Convert pixel range from [0,255]
    to [0,1].
    """

    image = image.astype("float32")

    return image / 255.0



def preprocess_image(image):

    """
    Complete preprocessing pipeline.
    """

    image = resize_image(image)

    image = normalize_image(image)

    return image
