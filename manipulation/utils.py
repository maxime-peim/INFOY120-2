from __future__ import annotations

from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup

# trunk-ignore(flake8/F401)
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .extract import extract
from .preprocess import preprocess

if TYPE_CHECKING:
    from pandas import DataFrame

RAW_DIR = "raw"

EXTRACTED_FILE = "extracted.csv"
PREPROCESSED_FILE = "preprocessed.csv"
LABELS_FILE = "labels.csv"

TRAINING_DATA = Path("data/TR")
TESTING_DATA = Path("data/TT")
ANALYSE_FOLDER = Path("data/analyse")

VECTORIZER = FeatureUnion(
    [
        (
            "content_tfidf",
            Pipeline(
                [
                    (
                        "extract_content",
                        FunctionTransformer(lambda x: x["content"], validate=False),
                    ),
                    ("tfidf", TfidfVectorizer()),
                ]
            ),
        ),
        (
            "subject_tfidf",
            Pipeline(
                [
                    (
                        "extract_subject",
                        FunctionTransformer(lambda x: x["subject"], validate=False),
                    ),
                    ("tfidf", TfidfVectorizer()),
                ]
            ),
        ),
        (
            "from_tfidf",
            Pipeline(
                [
                    (
                        "from_subject",
                        FunctionTransformer(lambda x: x["from"], validate=False),
                    ),
                    ("tfidf", TfidfVectorizer()),
                ]
            ),
        ),
    ],
    # ('email_length',
    # Pipeline([('content_length_dict',
    #             FunctionTransformer(lambda x: [{"lenght": l} for l in x['content_length']],
    #                                 validate=False)),
    #             ('vect',
    #              DictVectorizer())]))],
    n_jobs=-1,
)


def denoise_html(content: str) -> str:
    content = BeautifulSoup(content, "html.parser").get_text()
    return content


def extract_and_preprocess(
    data_folder: Path, force_extract=False, force_preprocess=False
) -> DataFrame:
    """Extract and preprocess emails.
    If already done in the past and force = false, the data are read from files stored on disk.

    Args:
        data_folder (pathlib.Path): folder containing a raw folder with all emails.
        force_extract (bool, optional): Set to true to redo the extraction if already done. Defaults to False.
        force_preprocess (bool, optional): Set to true to redo the preprocessing if already done. Defaults to False.

    Returns:
        pd.DataFrame: dataframe containing preprocessed emails and features.
    """
    extracted_df = extract(data_folder, force=force_extract)
    processed_df = preprocess(data_folder, extracted_df, force=force_preprocess)
    return processed_df


def get_data(
    training_folder: Path, testing_folder: Path, force=None
) -> tuple[DataFrame, DataFrame]:
    force_extract = "extract" in force if force is not None else False
    force_preprocess = "preprocess" in force if force is not None else False

    # get preprocessed df from training dataset
    train_df = extract_and_preprocess(
        training_folder, force_extract=force_extract, force_preprocess=force_preprocess
    )
    # get preprocessed df from testing dataset
    test_df = extract_and_preprocess(
        testing_folder, force_extract=force_extract, force_preprocess=force_preprocess
    )

    train_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)

    return train_df, test_df


def save_plot(filepath: Path, **kwargs):
    plt.savefig(filepath, **kwargs)
    plt.clf()


def generate_pipeline(clf):
    return Pipeline(
        [
            ("vect", VECTORIZER),
            ("clf", clf),
        ]
    )


def fit_and_predict(clf, X_train, y_train, X_test):
    print("_" * 80)
    print("Training: ")
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print(f"train time: {train_time:.03}s")

    print("_" * 80)
    print("Predicting: ")
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print(f"test time: {test_time:.03}s")

    pred_df = pd.DataFrame(pred, columns=["prediction"])
    pred_df.index.name = "id"
    pred_df.index += 1

    return pred_df, train_time, test_time
