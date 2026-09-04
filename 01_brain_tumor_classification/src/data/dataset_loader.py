# Brain MRI dataset loader.

import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from .preprocessing import preprocess_image



CLASS_NAMES = [

    "glioma",
    "meningioma",
    "pituitary",
    "no_tumor"

]



def load_dataset(dataset_path):

    """
    Load MRI images and labels.

    Returns:
        X : numpy array
        y : labels
    """


    images = []
    labels = []


    for label, class_name in enumerate(CLASS_NAMES):

        class_path = os.path.join(
            dataset_path,
            class_name
        )


        for filename in os.listdir(class_path):

            image_path = os.path.join(
                class_path,
                filename
            )


            image = cv2.imread(
                image_path
            )


            if image is None:
                continue


            image = preprocess_image(
                image
            )


            images.append(image)

            labels.append(label)



    X = np.array(
        images,
        dtype="float32"
    )


    y = np.array(
        labels
    )


    return X,y




def create_train_validation_split(
        X,
        y,
        validation_size=0.2,
        seed=42
):

    """
    Stratified 80/20 split.

    The paper does not specify
    random seed, therefore 42 is used.
    """


    return train_test_split(

        X,
        y,

        test_size=validation_size,

        random_state=seed,

        stratify=y

    )
