import os
import logging
from typing import Tuple

import pandas as pd
import torch

from net import model, criterion, optimizer
from common.minio_connector import MinioClass
from common.config import config
from common.logging_setup import configure_logger

logger = logging.getLogger(__name__)


def download_and_prepare_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Preparation function

    :return: train data and labels
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    client.m_client.fget_object(
        os.getenv("RUN_ID"),
        "data_prep/artifacts/train.csv",
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

    os.chdir(os.path.join(os.getenv("STAGE"), "artifacts", "checkpoints"))
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
                f"checkpoint-{i}.pth",
            )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model.pth",
    )
    logger.info("Training has been finished!")


def serialize_model() -> None:
    """Create serializable and optimizable models (TorchScript) from PyTorch code."""

    checkpoint = torch.load("model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()
    dummy_input = torch.FloatTensor([4.0, 3.3, 1.7, 0.5])
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, "model.pt")
    logger.info(
        "Serialized model has been saved to: %s",
        os.path.join(os.getcwd(), "model.pt"),
    )


def main() -> None:
    """Training main function."""

    torch.manual_seed(config["train_eval"].get("seed"))
    configure_logger(config["logging"])
    os.makedirs(
        os.path.join(os.getenv("STAGE"), "artifacts", "checkpoints"), exist_ok=True
    )
    x_train, y_train = download_and_prepare_data()
    train(config["train_eval"], x_train, y_train)
    serialize_model()
    client.push_data()


if __name__ == "__main__":
    client = MinioClass()
    main()
