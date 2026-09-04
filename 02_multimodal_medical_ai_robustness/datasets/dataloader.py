from torch.utils.data import DataLoader



def create_loader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=4
):


    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=True

    )
