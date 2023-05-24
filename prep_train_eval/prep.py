import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from common.minio_connector import MinioClass
from common.config import config
from common.logging_setup import configure_logger

logger = logging.getLogger(__name__)


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

    os.chdir(os.path.join(os.getenv("STAGE"), "artifacts"))
    train_data, test_data = train_test_split(
        data, test_size=conf.get("train_test_split_ratio"), random_state=42
    )
    train_data.to_csv("train.csv", encoding="utf-8", index=False)
    test_data.to_csv("test.csv", encoding="utf-8", index=False)


def main() -> None:
    """Data preparation main function."""

    configure_logger(config["logging"])
    os.makedirs(os.path.join(os.getenv("STAGE"), "artifacts"), exist_ok=True)
    dataset = clean_dataset(config["data_prep"])
    split_dataset(dataset, config["data_prep"])
    client.push_data()


if __name__ == "__main__":
    client = MinioClass()
    main()
