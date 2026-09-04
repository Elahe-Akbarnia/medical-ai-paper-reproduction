# Checkpoint utilities for saving and loading training states.


from pathlib import Path
import torch



def save_checkpoint(
        state,
        checkpoint_dir: str,
        filename: str = "checkpoint.pt"
):
    """
    Save training checkpoint.

    Args:
        state (dict):
            Dictionary containing:
            - epoch
            - model_state_dict
            - optimizer_state_dict
            - scheduler_state_dict
            - best_metric

        checkpoint_dir (str):
            Directory for saving checkpoints.

        filename (str):
            Checkpoint file name.
    """

    checkpoint_path = Path(checkpoint_dir)

    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True
    )


    file_path = checkpoint_path / filename


    torch.save(
        state,
        file_path
    )


    return str(file_path)



def load_checkpoint(
        checkpoint_path: str,
        model,
        optimizer=None,
        scheduler=None,
        device="cuda"
):
    """
    Load checkpoint and restore training state.

    Args:

        checkpoint_path:
            Path to checkpoint file.

        model:
            PyTorch model.

        optimizer:
            Optional optimizer.

        scheduler:
            Optional LR scheduler.

        device:
            CPU/GPU device.


    Returns:

        start_epoch,
        best_metric
    """


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )


    if scheduler is not None:

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )


    start_epoch = (
        checkpoint["epoch"] + 1
    )


    best_metric = checkpoint.get(
        "best_metric",
        None
    )


    return start_epoch, best_metric



def save_best_model(
        model,
        metric,
        best_metric,
        checkpoint_dir,
        filename="best_model.pt"
):
    """
    Save model only if performance improves.

    Example:
        Higher F1-score is better.
    """


    if best_metric is None or metric > best_metric:


        path = Path(checkpoint_dir)

        path.mkdir(
            parents=True,
            exist_ok=True
        )


        torch.save(
            model.state_dict(),
            path / filename
        )


        return metric, True


    return best_metric, False
