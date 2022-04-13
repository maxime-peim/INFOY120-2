from __future__ import annotations

import json
from datetime import datetime
from pprint import pprint
from time import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# trunk-ignore(flake8/F401)
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer

# trunk-ignore(flake8/F401)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import utils

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Union

    from pandas import DataFrame
    from sklearn.base import BaseEstimator

    Parameters = list[str, list[Any]]
    ParametersDict = dict[str, Parameters]
    ClassifierParameters = tuple[str, ParametersDict]
    ClassifierScore = tuple[str, float, float, float]


def benchmark(
    clf: BaseEstimator,
    parameters_names: list[str],
    output_folder: Path,
    X_train: np.array,
    y_train: Union[np.array, DataFrame],
    X_test: np.array,
) -> ClassifierScore:
    print("_" * 80)
    print("Training: ")

    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print(f"train time: {train_time:.03}s")

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print(f"test time: {test_time:.03}s")

    pred_df = pd.DataFrame(pred, columns=["prediction"])
    pred_df.index.name = "id"
    pred_df.index += 1

    all_parameters = clf.best_estimator_.get_params()
    optimal_parameters = {name: all_parameters[name] for name in parameters_names}

    clf_descr = all_parameters["clf"].__class__.__name__

    pprint(all_parameters)

    score = clf.best_score_
    formatted = pred_df.applymap(lambda x: int(x != "spam"))
    formatted.to_csv(output_folder.joinpath("tmp.csv"), index_label="id")

    clf_folder = output_folder.joinpath(clf_descr).joinpath(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    clf_folder.mkdir(exist_ok=True, parents=True)
    labels_file = clf_folder.joinpath(utils.LABELS_FILE)
    if labels_file.exists():
        labels_file.unlink()
    formatted.to_csv(labels_file, index_label="id")
    with clf_folder.joinpath("params.txt").open("w") as fp:
        json.dump(optimal_parameters, fp, indent=4, sort_keys=True)

    return clf_descr, score, train_time, test_time


def classify(
    classifiers: list[ClassifierParameters],
    training_folder: Path,
    testing_folder: Path,
    force=None,
    plot=False,
):

    force_extract = "extract" in force if force is not None else False
    # force preprocessing if forced extraction
    force_preprocess = (
        "preprocess" in force if force is not None else False
    ) or force_extract

    # get preprocessed df from training dataset
    train_df = utils.extract_and_preprocess(
        training_folder, force_extract=force_extract, force_preprocess=force_preprocess
    )
    # get preprocessed df from testing dataset
    test_df = utils.extract_and_preprocess(
        testing_folder, force_extract=force_extract, force_preprocess=force_preprocess
    )

    train_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)

    output_folder = testing_folder.joinpath("labels")
    output_folder.mkdir(exist_ok=True, parents=True)

    y_train = train_df.prediction

    results = []
    for clf, parameters in classifiers:

        vectorizer = FeatureUnion(
            [
                (
                    "content_tfidf",
                    Pipeline(
                        [
                            (
                                "extract_content",
                                FunctionTransformer(
                                    lambda x: x["content"], validate=False
                                ),
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
                                FunctionTransformer(
                                    lambda x: x["subject"], validate=False
                                ),
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
                                FunctionTransformer(
                                    lambda x: x["from"], validate=False
                                ),
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

        pipeline = Pipeline(
            [
                ("vect", vectorizer),
                ("clf", clf),
            ]
        )

        #        gs_vect = RandomizedSearchCV(
        #            pipeline,
        #            parameters,
        #            cv=10,
        #            n_jobs=-1,
        #            scoring=make_scorer(f1_score, pos_label="ham"),
        #            verbose=4,
        #            random_state=42,
        #            n_iter=1000,
        #        )

        gs_vect = GridSearchCV(
            pipeline,
            parameters,
            cv=10,
            n_jobs=-1,
            scoring=make_scorer(f1_score, pos_label="ham"),
            verbose=3,
        )

        print("=" * 80)
        print(clf.__class__.__name__)
        results.append(
            benchmark(
                gs_vect,
                list(parameters.keys()),
                output_folder,
                train_df,
                y_train,
                test_df,
            )
        )

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    best_clf_i = np.argmax(score)
    print(f"Best classifiers: {clf_names[best_clf_i]} -> {score[best_clf_i]}")

    if plot:
        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, 0.2, label="score", color="navy")
        plt.barh(indices + 0.3, training_time, 0.2, label="training time", color="c")
        plt.barh(indices + 0.6, test_time, 0.2, label="test time", color="darkorange")
        plt.yticks(())
        plt.legend(loc="best")
        plt.subplots_adjust(left=0.25)
        plt.subplots_adjust(top=0.95)
        plt.subplots_adjust(bottom=0.05)

        for i, c in zip(indices, clf_names):
            plt.text(-0.3, i, c)

        plt.show()
