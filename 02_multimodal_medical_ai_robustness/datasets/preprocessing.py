# Preprocessing utilities for multimodal medical dataset.

# Image: resize, tensor conversion and normalization

# Text: lowercase and whitespace cleaning


import re

from torchvision import transforms



def get_image_transform(
        image_size=224
):
    """
    Image preprocessing pipeline.

    Following paper:
    mean = 0.61
    std = 0.24
    """

    transform = transforms.Compose(
        [

            transforms.Resize(
                (image_size,image_size)
            ),


            transforms.ToTensor(),


            transforms.Normalize(
                mean=[0.61],
                std=[0.24]
            )

        ]
    )

    return transform




def clean_text(text:str)->str:
    """
    Clinical text preprocessing.

    Paper:
    - lowercase
    - remove extra whitespace
    """

    if text is None:
        return ""


    text=text.lower()


    text=re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()
