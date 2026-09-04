# Checkpoint utilities.


import os



def create_directory(path):


    if not os.path.exists(path):

        os.makedirs(path)



def get_checkpoint_path(

        name

):


    create_directory(

        "checkpoints"

    )


    return os.path.join(

        "checkpoints",

        name

    )
