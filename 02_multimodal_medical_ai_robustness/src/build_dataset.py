from datasets.indiana_dataset import (
    IndianaDatasetBuilder
)


from datasets.image_dataset import (
    XRayImageDataset
)


from datasets.text_dataset import (
    ClinicalTextDataset
)


from datasets.preprocessing import (
    get_image_transform
)



def prepare_dataframe(
        reports,
        projections,
        images
):


    builder=IndianaDatasetBuilder(

        reports,

        projections,

        images

    )


    return builder.build_dataframe()
