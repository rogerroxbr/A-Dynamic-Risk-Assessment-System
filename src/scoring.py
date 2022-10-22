"""
Author: Roger de Tarso Guerra
Date: October, 2022
This script used for scoring model
"""
import os
import sys
import glob
import time
import logging


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from utils import load_model
from config import MODEL_PATH, TEST_DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def prepare_data(dataset_path: str, dropped_columns: list = None) -> dict:
    """
    Prepares test dataset for use.
    :param dataset_path: Directory contained CSV dataset(s)
    :param dropped_columns: List of columns to be dropped from DataFrame
    :return:
    """
    dataset_list = glob.glob(f"{dataset_path}/*.csv")
    logging.info("Found %s files. Creating dataframe", len(dataset_list))

    df = pd.concat(map(pd.read_csv, dataset_list))
    if dropped_columns:
        df.drop(dropped_columns, axis=1, inplace=True)

    logging.info("Test dataset is of shape: %s", df.shape)

    y = df.pop("exited")
    data = {"test": {"X": df, "y": y}}
    return data


def score_model(
    data: dict,
    model: LogisticRegression,
    output_to_file: bool = True,
    metric_output_dir: str = None,
) -> float:
    """
    Use input model to make predictions on test data and calculate F1-Score
    :param data: Data for which predictions are to be mad.e
    :param model: Trained LogisticRegression model
    :param output_to_file: Whether to write F1-Score to file
    :param metric_output_dir: Directory where F1-Score is written to
    :return: None
    """
    predictions = model.predict(data["test"]["X"])
    logging.info(classification_report(data["test"]["y"], predictions))

    f1score = f1_score(data["test"]["y"], predictions)
    logging.info("Model F1-Score : %s", f1score)

    if output_to_file:
        if not metric_output_dir:
            raise Exception("metric_output_dir should not be None")
        os.makedirs(metric_output_dir, exist_ok=True)
        metric_file_path = os.path.join(
            metric_output_dir,
            f"latestscore_{time.strftime('%y%m%d%H%M%S')}.txt",
        )
        with open(metric_file_path, "w") as file:
            logging.info("Writing F1-Score to %s", metric_file_path)
            file.write(str(f1score))
    return f1score


if __name__ == "__main__":
    model_path = MODEL_PATH
    test_data_path = TEST_DATA_PATH
    data = prepare_data(test_data_path, dropped_columns=["corporation"])
    model = load_model(model_path)
    score = score_model(data, model, metric_output_dir=model_path)

    if model is None:
        raise Exception(f"No model found in {model_path}")

    logging.info(score)
