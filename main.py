import argparse
from pathlib import Path

# trunk-ignore(flake8/F401)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# trunk-ignore(flake8/F401)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# trunk-ignore(flake8/F401)
from sklearn.feature_selection import SelectFromModel

# trunk-ignore(flake8/F401)
from sklearn.gaussian_process import GaussianProcessClassifier

# trunk-ignore(flake8/F401)
from sklearn.gaussian_process.kernels import RBF

# trunk-ignore(flake8/F401)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)

# trunk-ignore(flake8/F401)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB

# trunk-ignore(flake8/F401)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# trunk-ignore(flake8/F401)
from sklearn.neural_network import MLPClassifier

# trunk-ignore(flake8/F401)
from sklearn.pipeline import Pipeline

# trunk-ignore(flake8/F401)
from sklearn.svm import SVC, LinearSVC, OneClassSVM

# trunk-ignore(flake8/F401)
from sklearn.tree import DecisionTreeClassifier

import manipulation.utils as utils
from manipulation import analyse, classify, extract, preprocess

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
    ]
    """
    classifiers = [
        (DecisionTreeClassifier(), {
            'clf__max_depth': range(10, 110, 10),
            'vect__content_tfidf__tfidf__max_features': range(100, 1000, 100),
            'vect__subject_tfidf__tfidf__max_features': range(10, 110, 10),
        })
    ]
    classifiers = [
        (OneClassSVM(), {
            'clf__kernel' : ['rbf'],
            'clf__gamma' : [0.001, 0.01, 0.1, 1],
            'clf__nu': [0.001, 0.01, 0.1, 1],
            'vect__content_tfidf__tfidf__max_features': range(100, 1000, 100),
            'vect__subject_tfidf__tfidf__max_features': range(10, 110, 10),
        })
    ]"""
    """
    classifiers = [
        (LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0), {
            "clf__C": np.linspace(0, 2., 100)[1:],
            "clf__penalty" : ['l2', 'l1'],
            'vect__content_tfidf__tfidf__max_features': range(100, 1000, 100),
            'vect__subject_tfidf__tfidf__max_features': range(10, 110, 10),
        })
    ]
    """

    if all_actions or "extract" in actions:
        extract(args.training_folder, force=True)
        extract(args.testing_folder, force=True)

    if all_actions or "preprocess" in actions:
        preprocess(args.training_folder, force=True)
        preprocess(args.testing_folder, force=True)

    if all_actions or "classify" in actions:
        classify(classifiers, args.training_folder, args.testing_folder, plot=args.plot)

    if all_actions or "analyse" in actions:
        analyse(args.training_folder, args.testing_folder, args.analyse_folder)
