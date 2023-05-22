import os
import logging
import sys
import json

import torch
from flask import Flask, jsonify, request
from minio import Minio
from minio.error import S3Error

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FORMATTER = logging.Formatter(LOG_FORMAT)
logger = logging.getLogger()
logger.setLevel("INFO")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(FORMATTER)
logger.addHandler(stdout_handler)

app = Flask(__name__)


def download_model(conf: dict) -> None:
    """Download serialized model from object storage

    :param conf: inference configuration
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

    client.fget_object(
        os.getenv("RUN_ID"),
        "train-eval/artifacts/train/model.pt",
        conf.get("model_path"),
    )


def labels_to_classes_mapping(conf: dict) -> dict:
    """Generate the labels to classes mapping

    :param conf: full configuration
    :type conf: dict
    :return classes dictionary
    :rtype: dict
    """

    mapping = {v: k for k, v in conf["data_prep"].get("mapping").items()}
    return mapping


def get_prediction(input_data: list):
    """Run model on the input data

    :param input_data: provided data
    :type input_data: list
    """

    input_tensor = torch.tensor(input_data)
    class_id = model(input_tensor).argmax().item()
    class_name = labels[class_id]
    return class_id, class_name


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        json_data = request.json["data"]
        class_id, class_name = get_prediction(json_data)
        return jsonify({"class_id": class_id, "class_name": class_name})


if __name__ == "__main__":
    if os.getenv("CONFIG_FILE_PATH") is None:
        logger.error("Configuration file path is missing!")
        sys.exit(0)
    else:
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as json_file:
            config = json.load(json_file)
    download_model(config["inference"])
    model = torch.jit.load(
        config["inference"].get("model_path"),
    )
    labels = labels_to_classes_mapping(config)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
