from __future__ import annotations

import json
from datetime import datetime
from pprint import pprint
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, make_scorer

# trunk-ignore(flake8/F401)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from . import utils

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Union

    from numpy import array as NumpyArray
    from pandas import DataFrame
    from sklearn.base import BaseEstimator

    Parameters = list[str, list[Any]]
    ParametersDict = dict[str, Parameters]
    ClassifierParameters = tuple[str, ParametersDict]
    ClassifierScore = tuple[str, float, float, float]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def benchmark_classifier(
    clf: BaseEstimator,
    parameters_names: list[str],
    output_folder: Path,
    X_train: DataFrame,
    y_train: Union[NumpyArray, DataFrame],
    X_test: DataFrame,
) -> ClassifierScore:
    pred_df, train_time, test_time = utils.fit_and_predict(
        clf, X_train, y_train, X_test
    )

    all_parameters = clf.best_estimator_.get_params()
    optimal_parameters = {name: all_parameters[name] for name in parameters_names}

    clf_descr = all_parameters["clf"].__class__.__name__

    pprint(all_parameters)

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

    return clf_descr, clf.best_score_, train_time, test_time


def classify(
    classifiers: list[ClassifierParameters],
    training_folder: Path,
    testing_folder: Path,
    plot=False,
):
    output_folder = testing_folder.joinpath("labels")
    output_folder.mkdir(exist_ok=True, parents=True)

    train_df, test_df = utils.get_data(training_folder, testing_folder)
    y_train = train_df.prediction

    results = []
    for clf, parameters in classifiers:

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

        pipeline = utils.generate_pipeline(clf)

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
            utils.benchmark_classifier(
                gs_vect,
                list(parameters.keys()),
                output_folder,
                train_df,
                y_train,
                test_df,
            )
        )

        pprint(gs_vect.cv_results_)
        with open("data/analyse/rf.json", "w") as outfile:
            json.dump(gs_vect.cv_results_, outfile, indent=4, cls=NumpyEncoder)

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
