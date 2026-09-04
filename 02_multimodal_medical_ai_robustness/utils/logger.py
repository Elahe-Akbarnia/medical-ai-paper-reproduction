import logging
from pathlib import Path


def create_logger(
        name: str,
        log_file: str
):

    Path(log_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)


    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )


    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)


    return logger
