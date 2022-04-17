import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import manipulation as mpl
import manipulation.utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "actions",
        nargs="*",
        choices=["classify", "analyse", "extract", "preprocess", "all"],
        default="all",
    )
    parser.add_argument("--training-folder", type=Path, default=utils.TRAINING_DATA)
    parser.add_argument("--testing-folder", type=Path, default=utils.TESTING_DATA)
    parser.add_argument("--analyse-folder", type=Path, default=utils.ANALYSE_FOLDER)
    parser.add_argument("-p", "--plot", action="store_true")

    args = parser.parse_args()
    actions = args.actions
    all_actions = "all" in actions

    # we fix the random state of each classifier, so we can reproduce
    # the exact results for a given set of parameters

    # clf__ -> classifier's parameters
    # vect__content_tfidf__ -> body content TFIDF vectorizer's parameters
    # vect__subject_tfidf__ -> subject TFIDF vectorizer's parameters
    classifiers = [
        (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                #'clf__bootstrap': [True, False],
                "clf__max_depth": [None, *list(range(10, 110, 10))],
                #'clf__max_features': ['auto', 'sqrt'],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__min_samples_split": [2, 5, 10],
                "clf__n_estimators": range(100, 190, 10),
                "vect__content_tfidf__tfidf__max_features": range(600, 1200, 100),
                "vect__subject_tfidf__tfidf__max_features": range(30, 130, 10),
            },
        ),
        (
            DecisionTreeClassifier(random_state=42),
            {
                "clf__max_depth": range(10, 110, 10),
                "vect__content_tfidf__tfidf__max_features": range(100, 1000, 100),
                "vect__subject_tfidf__tfidf__max_features": range(10, 110, 10),
            },
        ),
        (
            SVC(random_state=42),
            {
                "clf__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                "clf__degree": range(1, 10),
                "vect__content_tfidf__tfidf__max_features": range(100, 1000, 100),
                "vect__subject_tfidf__tfidf__max_features": range(10, 110, 10),
            },
        ),
    ]

    if all_actions or "extract" in actions:
        mpl.extract(args.training_folder, force=True)
        mpl.extract(args.testing_folder, force=True)

    if all_actions or "preprocess" in actions:
        mpl.preprocess(args.training_folder, force=True)
        mpl.preprocess(args.testing_folder, force=True)

    if all_actions or "classify" in actions:
        mpl.classify(
            classifiers, args.training_folder, args.testing_folder, plot=args.plot
        )

    if all_actions or "analyse" in actions:
        mpl.analyse(args.training_folder, args.testing_folder, args.analyse_folder)
