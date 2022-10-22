"""
Author: Roger de Tarso Guerra
Date: October, 2022
This script used to deploy the trained model
"""
import os
import shutil
import sys
import logging

from utils import get_latest_file
from config import MODEL_PATH, DATA_PATH, PROD_DEPLOYMENT_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def store_production_files(dst: str, *args) -> None:
    """
    Copy files from their source directories to a destination dir
    :param dst: Destination dir to copy files
    :param args: Files to be be copied
    :return: None
    """
    os.makedirs(dst, exist_ok=True)
    for file in args:
        filename = file.split(os.sep)[1]
        new_filename = filename.split("_")[0] + "." + filename.split(".")[-1]
        path = os.path.join(dst, new_filename)
        logging.info("Copying %s to %s", filename, path)
        shutil.copy2(file, path)


if __name__ == "__main__":
    model_dir = MODEL_PATH
    output_folder_path = DATA_PATH
    deployment_path = PROD_DEPLOYMENT_PATH

    model_path = get_latest_file(model_dir, "trainedmodel_*.pkl")
    metric_path = get_latest_file(model_dir, "latestscore_*.txt")
    ingest_record_path = get_latest_file(
        output_folder_path, "ingestedfiles_*.txt"
    )
    store_production_files(
        deployment_path, model_path, metric_path, ingest_record_path
    )
