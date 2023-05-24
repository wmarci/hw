import os
import logging
from typing import Tuple, List

import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from net import model, optimizer
from common.minio_connector import MinioClass
from common.config import config
from common.logging_setup import configure_logger
from common.util import MAPPING

logger = logging.getLogger(__name__)


def download_and_prepare_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Preparation function

    :return: test data and labels
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    client.m_client.fget_object(
        os.getenv("RUN_ID"),
        "data_prep/artifacts/test.csv",
        "./data/test.csv",
    )
    client.m_client.fget_object(
        os.getenv("RUN_ID"),
        "train/artifacts/checkpoints/model.pth",
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


def compute_metrics(true_labels: List[float], predicted_labels: List[float]) -> None:
    """Calculate evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)

    :param true_labels: list of ground truth labels
    :type true_labels: List[float]
    :param predicted_labels: list of predicted labels
    :type predicted_labels: List[float]
    """

    os.chdir(os.path.join(os.getenv("STAGE"), "artifacts"))
    true_classes = [MAPPING[item] for item in true_labels]
    predicted_classes = [MAPPING[item] for item in predicted_labels]
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
        "metrics.csv",
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
    plt.savefig("CM.png")


def main() -> None:
    """Training main function."""

    configure_logger(config["logging"])
    os.makedirs(os.path.join(os.getenv("STAGE"), "artifacts"), exist_ok=True)
    x_test, y_test = download_and_prepare_data()
    y_list, y_hat_list = test(x_test, y_test)
    compute_metrics(y_list, y_hat_list)
    client.push_data()


if __name__ == "__main__":
    client = MinioClass()
    main()
