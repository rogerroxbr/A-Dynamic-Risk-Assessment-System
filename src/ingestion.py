"""
Author: Roger de Tarso Guerra
Date: October, 2022
This script used for ingesting data
"""
import glob
import os
import time
import logging
import sys
import pandas as pd
from pandas import DataFrame
from config import INPUT_FOLDER_PATH, DATA_PATH

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def merge_multiple_dataframe(input_dir: str, output_dir: str) -> DataFrame:
    """
    Reads multiple CSV files into a pandas dataframe.
    :param input_dir: path to directory containing CSV files
    :param output_dir: list of ingested files are written to {output_dir}/ingestedfiles_*.txt
    :return: DataFrame containing all CSV datasets found in path
    """

    datasets = glob.glob(f"{input_dir}/*.csv")
    logging.info(f"Found {len(datasets)} files. Creating dataframe")

    df = pd.concat(map(pd.read_csv, datasets))

    os.makedirs(output_dir, exist_ok=True)
    output_path = (
        f"{output_dir}/ingestedfiles_{time.strftime('%y%m%d%H%M%S')}.txt"
    )
    with open(output_path, "w") as file:
        print(f"Writing list of ingested files to {output_path}")
        file.write("\n".join(datasets))

    return df


def clean_dataset(df: DataFrame) -> DataFrame:
    """
    All cleaning operations are done here.
    - Remove duplicate rows
    :param df: Input DataFrame to be cleaned
    :return: Cleaned DataFrame
    """
    temp = df.copy(deep=True)
    logging.info(f"Input DataFrame is of shape: {temp.shape}")
    print("Missing Values", temp.isna().sum(), sep="\n")

    # Drop duplicate columns
    temp.drop_duplicates(keep="first", ignore_index=True, inplace=True)
    logging.info(
        f"Dropped duplicate rows. DataFrame is of shape: {temp.shape}"
    )
    return temp


if __name__ == "__main__":

    input_folder_path = INPUT_FOLDER_PATH
    output_folder_path = DATA_PATH

    concat_df = merge_multiple_dataframe(input_folder_path, output_folder_path)
    cleaned_df = clean_dataset(concat_df)

    output_df_path = os.path.join(
        output_folder_path, f"finaldata_{time.strftime('%y%m%d%H%M%S')}.csv"
    )
    logging.info(f"Writing cleaned DataFrame to {output_df_path}")
    cleaned_df.to_csv(output_df_path, index=False)
