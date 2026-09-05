from torchvision import transforms



def get_image_transform(
        image_size=224
):

    transform = transforms.Compose(
        [

            transforms.Resize(
                (image_size,image_size)
            ),


            transforms.ToTensor(),


            transforms.Normalize(
                mean=[0.61,0.61,0.61],
                std=[0.24,0.24,0.24]
            )

        ]
    )


    return transform



def clean_text(text):

    if text is None:
        return ""

    text = text.lower()

    return " ".join(
        text.split()
    )
