
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader


from datasets.indiana_dataset import IndianaDatasetBuilder

from datasets.image_dataset import XRayImageDataset

from datasets.preprocessing import get_image_transform



def prepare_dataframe(
        reports,
        projections,
        images
):

    builder = IndianaDatasetBuilder(
        reports,
        projections,
        images
    )

    return builder.build_dataframe()



def build_image_loaders(
        dataframe,
        image_size=224,
        batch_size=8
):


    train_df, val_df = train_test_split(

        dataframe,

        test_size=0.2,

        random_state=42,

        stratify=dataframe["label"]

    )


    transform = get_image_transform(
        image_size
    )


    train_dataset = XRayImageDataset(

        train_df,

        transform=transform

    )


    val_dataset = XRayImageDataset(

        val_df,

        transform=transform

    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=2,

        pin_memory=True

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=2,

        pin_memory=True

    )


    return train_loader, val_loader
