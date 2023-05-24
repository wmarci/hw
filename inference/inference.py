import os
from typing import Tuple

import torch
from flask import Flask, jsonify, request

from common.minio_connector import MinioClass
from common.config import config
from common.logging_setup import configure_logger
from common.util import MAPPING

app = Flask(__name__)


def download_model(conf: dict) -> None:
    """Download serialized model from object storage

    :param conf: inference configuration
    :type conf: dict
    """

    client.m_client.fget_object(
        os.getenv("RUN_ID"),
        "train/artifacts/checkpoints/model.pt",
        conf.get("model_path"),
    )


def get_prediction(input_data: list) -> Tuple[int, str]:
    """Run model on the input data

    :param input_data: provided data
    :type input_data: list
    :return: label and the class name
    :rtype: Tuple[int, str]
    """

    input_tensor = torch.tensor(input_data)
    class_id = model(input_tensor).argmax().item()
    class_name = MAPPING[class_id]

    return class_id, class_name


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        json_data = request.json["data"]
        class_id, class_name = get_prediction(json_data)

        return jsonify({"class_id": class_id, "class_name": class_name})


if __name__ == "__main__":
    configure_logger(config["logging"])
    client = MinioClass()
    download_model(config["inference"])
    model = torch.jit.load(
        config["inference"].get("model_path"),
    )
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
