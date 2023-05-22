import os
import logging
import sys
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from minio import Minio

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


def clean_dataset(conf: dict) -> pd.core.frame.DataFrame:
    """Eliminate duplicated records from the dataset, if there are any.

    :param conf: configuration of the data_prep stage
    :type conf: dict
    :return: cleaned dataset as dataframe
    :rtype: pd.core.frame.DataFrame
    """

    dataset_df = pd.read_csv(conf.get("dataset_url"), header=None)
    logger.info("Dataset shape: %s", dataset_df.shape)
    duplicated_records = dataset_df.duplicated()
    if duplicated_records.any():
        logger.warning("Dataset is not clean, duplicated records have been found!")
        logger.debug("Duplicated records: %s", dataset_df[duplicated_records])
        dataset_df.drop_duplicates(inplace=True)
        logger.info("Duplicated records have been eliminated!")
        logger.info("Dataset shape: %s", dataset_df.shape)
    else:
        logger.info("Dataset is clean!")

    dataset_df.columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
    ]
    dataset_df["labels"] = dataset_df["species"].apply(lambda x: conf.get("mapping")[x])
    logger.info("Mapping has been applied!")

    return dataset_df


def split_dataset(data: pd.core.frame.DataFrame, conf: dict) -> None:
    """Split the dataset into train and test datasets.

    :param data: dataset
    :type data: pd.core.frame.DataFrame
    :param conf: configuration of the data_prep stage
    :type conf: dict
    """

    train_data, test_data = train_test_split(
        data, test_size=conf.get("train_test_split_ratio"), random_state=42
    )
    train_data.to_csv("train.csv", encoding="utf-8", index=False)
    test_data.to_csv("test.csv", encoding="utf-8", index=False)


def push_data_to_minio() -> None:
    """Push train,test datasets, container log and configuration to Minio."""

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
    client.fput_object(
        os.getenv("RUN_ID"),
        "data_prep/artifacts/train/train.csv",
        "train.csv",
    )
    client.fput_object(
        os.getenv("RUN_ID"),
        "data_prep/artifacts/test/test.csv",
        "test.csv",
    )
    client.fput_object(
        os.getenv("RUN_ID"),
        "data_prep/logs/container.log",
        "container.log",
    )
    client.fput_object(
        os.getenv("RUN_ID"),
        "config/config.json",
        "/config/config.json",
    )
    logger.info("Upload successful!")


def main(conf: dict) -> None:
    """Data preparation main function

    :param conf: configuration of the data_prep stage
    :type conf: dict
    """

    dataset = clean_dataset(conf)
    split_dataset(dataset, conf)
    push_data_to_minio()


if __name__ == "__main__":
    if os.getenv("CONFIG_FILE_PATH") is None:
        logger.error("Configuration file path is missing!")
        sys.exit(0)
    else:
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as json_file:
            config = json.load(json_file)["data_prep"]
        main(config)

# df[df.duplicated(keep=False)]
