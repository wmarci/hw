import os
import logging
import sys
import json
from typing import Tuple

import pandas as pd
import torch
from minio import Minio
from minio.error import S3Error

from net import model, criterion, optimizer

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FORMATTER = logging.Formatter(LOG_FORMAT)

logger = logging.getLogger()
logger.setLevel("INFO")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(FORMATTER)
file_handler = logging.FileHandler("container.log")
file_handler.setFormatter(FORMATTER)
logger.addHandler(stdout_handler)
logger.addHandler(file_handler)


def download_and_prepare_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Preparation function

    :return: train data and labels
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    client = Minio(
        endpoint=os.getenv("MINIO_ADDRESS"),
        secure=False,
        access_key=os.getenv("MINIO_ROOT_USER"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
    )
    logger.info("Connected to Minio object storage.")
    try:
        client.bucket_exists(os.getenv("RUN_ID"))
    except S3Error as exc:
        logger.error("Bucket does not exist!")
        raise exc

    client.fget_object(
        os.getenv("RUN_ID"),
        "data_prep/artifacts/train/train.csv",
        "./data/train.csv",
    )
    dataset = pd.read_csv("./data/train.csv")
    x = dataset.drop(["species", "labels"], axis=1).values
    y = dataset["labels"].values
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)

    return x, y


def train(conf: dict, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
    """Train function

    :param conf: configuration of the train and eval stages
    :type conf: dict
    :param x_train: train data
    :type x_train: torch.Tensor
    :param y_train: train labels
    :type y_train: torch.Tensor
    """

    losses = []
    for i in range(conf.get("epochs")):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        logger.info(f"epoch: {i:2} loss: {loss.item():10.8f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % conf.get("checkpoints_save_iter") == 0:
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                os.path.join(conf.get("checkpoints_dir"), f"checkpoint-{i}.pth"),
            )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(conf.get("checkpoints_dir"), "model.pth"),
    )
    logger.info("Training has been finished!")


def serialize_model(conf: dict) -> None:
    """Create serializable and optimizable models (TorchScript) from PyTorch code

    :param conf: configuration of the train and eval stages
    :type conf: dict
    """

    checkpoint = torch.load(os.path.join(conf.get("checkpoints_dir"), "model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()
    dummy_input = torch.FloatTensor([4.0, 3.3, 1.7, 0.5])
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, os.path.join(conf.get("checkpoints_dir"), "model.pt"))
    logger.info(
        "Serialized model has been saved to: %s",
        os.path.join(conf.get("checkpoints_dir"), "model.pth"),
    )


def push_data_to_minio(conf: dict) -> None:
    """Push checkpoints and the container log to Minio.

    :param conf: configuration of the train and eval stages
    :type conf: dict
    """

    client = Minio(
        endpoint=os.getenv("MINIO_ADDRESS"),
        secure=False,
        access_key=os.getenv("MINIO_ROOT_USER"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
    )
    logger.info("Connected to Minio object storage.")
    found = client.bucket_exists(os.getenv("RUN_ID"))
    if not found:
        client.make_bucket(os.getenv("RUN_ID"))
    else:
        logger.info("Bucket: %s already exists!", os.getenv("RUN_ID"))

    logger.info("Uploading new files to Minio...")
    for file in os.listdir(conf.get("checkpoints_dir")):
        client.fput_object(
            os.getenv("RUN_ID"),
            f"train-eval/artifacts/train/{file}",
            f"{os.path.join(conf.get('checkpoints_dir'), file)}",
        )
    client.fput_object(
        os.getenv("RUN_ID"),
        "train-eval/logs/train/container.log",
        "container.log",
    )
    logger.info("Upload successful!")


def main(conf: dict) -> None:
    """Training main function

    :param conf: configuration of the train and eval stages
    :type conf: dict
    """

    torch.manual_seed(42)
    if not os.path.exists(conf.get("checkpoints_dir")):
        os.makedirs(conf.get("checkpoints_dir"), exist_ok=True)
    x_train, y_train = download_and_prepare_data()
    train(conf, x_train, y_train)
    serialize_model(conf)
    push_data_to_minio(conf)


if __name__ == "__main__":
    if os.getenv("CONFIG_FILE_PATH") is None:
        logger.error("Configuration file path is missing!")
        sys.exit(0)
    else:
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as json_file:
            config = json.load(json_file)["train_eval"]
        main(config)
