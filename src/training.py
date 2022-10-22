"""
Author: Roger de Tarso Guerra
Date: October, 2022
This script used for training and evaluation model
"""

import glob
import os
import pickle
import time
import sys
import logging
from typing import Union

import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import DATA_PATH, MODEL_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def prepare_dataset(
    dataset_path: str,
    val_size: float = 0.1,
    create_val_data: bool = True,
    dropped_columns: list = None,
) -> Union[dict, DataFrame]:

    """
    Reads the most recent dataset 'finaldata_*.csv' from
    the dataset_path and prepares it for training
    :param dataset_path: Directory containing CSV datasets named 'finaldata_*.csv'
    :param val_size: test_size to use in train_test_split when creating validation data
    :param create_val_data:
    :param dropped_columns: Columns to drop from dataset
    :return:
    if create_val_data:
        {
            'training: {'X': <x_train_df>: 'y': <y_train_df>},
            'val': {'X': <x_val_df', 'y': <y_val_df>}
        }
    else DataFrame
    """
    dataset_list = glob.glob(f"{dataset_path}/finaldata.csv")
    dataset_list.sort()
    dataset = pd.read_csv(dataset_list[-1])  # Most recent dataset is used.
    logging.info(
        "DataFrame was successfully created from %s", dataset_list[-1]
    )

    dataset.drop(dropped_columns, axis=1, inplace=True)
    logging.info("Dropped columns: %s", dropped_columns)

    if create_val_data:
        x_train, x_val = train_test_split(
            dataset, test_size=val_size, random_state=42
        )
        y_train, y_val = x_train.pop("exited"), x_val.pop("exited")
        data = {
            "training": {"X": x_train, "y": y_train},
            "val": {"X": x_val, "y": y_val},
        }
        logging.info(
            f"{val_size*100}% of dataset is held out as validation data"
        )

        return data
    return dataset


def train_model(data: dict, model_dir: str) -> None:
    """
    Fits model on input data, calculates performance metrics and
    dumps model to dir
    :param data: Data dictionary containing
    :param model_dir: Path to dir containing model
    :return: None
    """
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )
    logging.info("Fitting model...")

    model.fit(data["training"]["X"], data["training"]["y"])

    accuracy = model.score(data["val"]["X"], data["val"]["y"])
    logging.info("Model Accuracy: %s", accuracy)

    predictions = model.predict(data["val"]["X"])
    f1score = f1_score(data["val"]["y"], predictions)

    logging.info("Model F1-Score: %s", f1score)

    logging.info(classification_report(data["val"]["y"], predictions))

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"trainedmodel.pkl")

    logging.info("Persisting fitted model to %s ... ", model_path)

    with open(model_path, "wb") as modelfile:
        pickle.dump(model, modelfile)


if __name__ == "__main__":

    dataset_csv_path = DATA_PATH
    model_dir = MODEL_PATH
    dropped_columns = ["corporation"]
    VAL_SIZE = 0.1

    data = prepare_dataset(
        dataset_csv_path,
        dropped_columns=dropped_columns,
        val_size=VAL_SIZE,
        create_val_data=True,
    )
    train_model(data, model_dir=model_dir)
