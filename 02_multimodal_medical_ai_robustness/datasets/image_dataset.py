
import torch

from torch.utils.data import Dataset

from PIL import Image



class XRayImageDataset(Dataset):


    def __init__(
            self,
            dataframe,
            transform=None
    ):

        self.data = dataframe

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(
            self,
            index
    ):

        item = self.data.iloc[index]


        frontal = Image.open(
            item.frontal_image
        ).convert("RGB")


        lateral = Image.open(
            item.lateral_image
        ).convert("RGB")



        if self.transform:

            frontal = self.transform(
                frontal
            )

            lateral = self.transform(
                lateral
            )



        label = torch.tensor(
            item.label,
            dtype=torch.float32
        )


        return {

            "frontal": frontal,

            "lateral": lateral,

            "label": label

        }
