import os
import logging

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class MinioClass:
    def __init__(self):
        self.m_client = Minio(
            endpoint=os.getenv("MINIO_ADDRESS"),
            secure=False,
            access_key=os.getenv("MINIO_ROOT_USER"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
        )
        self.check_bucket()

    def check_bucket(self) -> None:
        """Check, if the required bucket exists"""

        logger.info("Connected to Minio object storage.")
        found = self.m_client.bucket_exists(os.getenv("RUN_ID"))
        if not found:
            if os.getenv("STAGE") == "data_prep":
                self.m_client.make_bucket(os.getenv("RUN_ID"))
                logger.info("Bucket does not exist! Bucket will be created.")
            else:
                logger.error("Bucket does not exist!")
                raise S3Error
        else:
            logger.info("Bucket: %s exists!", os.getenv("RUN_ID"))

    def push_data(self) -> None:
        """Push resulted artifacts and logs to object storage."""

        logger.info("Uploading new files to Minio...")
        relative_path = "/container_files"
        for path, _, files in os.walk(os.path.join(relative_path, os.getenv("STAGE"))):
            for name in files:
                self.m_client.fput_object(
                    os.getenv("RUN_ID"),
                    os.path.join(os.path.relpath(path, relative_path), name),
                    os.path.join(path, name),
                )
        logger.info("Upload successful!")
