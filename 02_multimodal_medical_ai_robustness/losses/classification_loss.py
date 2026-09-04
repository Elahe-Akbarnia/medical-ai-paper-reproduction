import torch.nn as nn



def get_loss():

    """
    Paper uses BinaryCrossEntropyLoss
    for binary medical classification.
    """

    return nn.BCEWithLogitsLoss()
