import os
import logging
import sys
import json
from typing import Tuple, List

import pandas as pd
import torch
from sklearn import metrics
from minio import Minio
from minio.error import S3Error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from net import model, optimizer

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

    :return: test data and labels
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
        "data_prep/artifacts/test/test.csv",
        "./data/test.csv",
    )
    client.fget_object(
        os.getenv("RUN_ID"),
        "train-eval/artifacts/train/model.pth",
        "./data/model.pth",
    )
    dataset = pd.read_csv("./data/test.csv")
    x = dataset.drop(["species", "labels"], axis=1).values
    y = dataset["labels"].values
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)

    return x, y


def test(x_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[List[float], List[float]]:
    """Train function

    :param x_test: test data
    :type x_test: torch.Tensor
    :param y_test: test labels
    :type y_test: torch.Tensor
    :return truth and predicted labels
    :rtype: Tuple[List[float], List[float]]
    """

    y_hat_list = []
    y_list = []
    checkpoint = torch.load("/container_files/data/model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()
    with torch.no_grad():
        for val, y in zip(x_test, y_test):
            y_hat = model(val)
            y_list.append(y.item())
            y_hat_list.append(y_hat.argmax().item())
    logger.debug("Predictions: %s", y_hat_list)
    logger.debug("Truth: %s", y_list)

    return y_list, y_hat_list


def compute_metrics(
    conf: dict, true_labels: List[float], predicted_labels: List[float]
) -> None:
    """Calculate evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)

    :param conf: full configuration
    :type conf: dict
    :param true_labels: list of ground truth labels
    :type true_labels: List[float]
    :param predicted_labels: list of predicted labels
    :type predicted_labels: List[float]
    """

    labels_to_classes_mapping = {
        v: k for k, v in conf["data_prep"].get("mapping").items()
    }
    true_classes = [labels_to_classes_mapping[item] for item in true_labels]
    predicted_classes = [labels_to_classes_mapping[item] for item in predicted_labels]
    logger.info("Computing classification metrics.")
    accuracy = metrics.accuracy_score(true_classes, predicted_classes)
    macro_averaged_precision = metrics.precision_score(
        true_classes, predicted_classes, average="macro"
    )
    macro_averaged_recall = metrics.recall_score(
        true_classes, predicted_classes, average="macro"
    )
    macro_averaged_f1_score = metrics.f1_score(
        true_classes, predicted_classes, average="macro"
    )
    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": macro_averaged_precision,
        "Recall": macro_averaged_recall,
        "F1-score": macro_averaged_f1_score,
    }
    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    metrics_df.to_csv(
        os.path.join(conf["train_eval"].get("metrics_dir"), "metrics.csv"),
        encoding="utf-8",
        index=False,
    )
    plt.figure(figsize=(18, 8))
    sns.heatmap(
        metrics.confusion_matrix(true_classes, predicted_classes),
        annot=True,
        xticklabels=np.unique(true_classes),
        yticklabels=np.unique(true_classes),
        cmap="summer",
    )
    plt.xlabel("Predicted Classes")
    plt.ylabel("True Classes")
    plt.savefig(os.path.join(conf["train_eval"].get("metrics_dir"), "CM.png"))


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
    try:
        client.bucket_exists(os.getenv("RUN_ID"))
    except S3Error as exc:
        logger.error("Bucket does not exist!")
        raise exc

    logger.info("Uploading new files to Minio...")
    for file in os.listdir(conf.get("metrics_dir")):
        client.fput_object(
            os.getenv("RUN_ID"),
            f"train-eval/artifacts/test/{file}",
            f"{os.path.join(conf.get('metrics_dir'), file)}",
        )
    client.fput_object(
        os.getenv("RUN_ID"),
        "train-eval/logs/test/container.log",
        "container.log",
    )
    logger.info("Upload successful!")


def main(conf: dict) -> None:
    """Training main function

    :param conf: configuration of the train and eval stages
    :type conf: dict
    """

    torch.manual_seed(42)
    if not os.path.exists(conf["train_eval"].get("metrics_dir")):
        os.makedirs(conf["train_eval"].get("metrics_dir"), exist_ok=True)
    x_test, y_test = download_and_prepare_data()
    y_list, y_hat_list = test(x_test, y_test)
    compute_metrics(conf, y_list, y_hat_list)
    push_data_to_minio(conf["train_eval"])


if __name__ == "__main__":
    if os.getenv("CONFIG_FILE_PATH") is None:
        logger.error("Configuration file path is missing!")
        sys.exit(0)
    else:
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as json_file:
            config = json.load(json_file)
        main(config)
