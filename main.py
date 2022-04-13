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
from manipulation import classify

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-folder", type=Path, default=utils.TRAINING_DATA)
    parser.add_argument("--testing-folder", type=Path, default=utils.TESTING_DATA)
    parser.add_argument("-f", "--force", nargs="*", choices=["extract", "preprocess"])
    parser.add_argument("-p", "--plot", action="store_true")

    args = parser.parse_args()

    classifiers = [
        (
            RandomForestClassifier(random_state=42),
            {
                #'clf__bootstrap': [True, False],
                "clf__max_depth": range(10, 110, 10),
                #'clf__max_features': ['auto', 'sqrt'],
                #'clf__min_samples_leaf': [1, 2, 4],
                #'clf__min_samples_split': [2, 5, 10],
                "clf__n_estimators": range(10, 110, 10),
                "vect__content_tfidf__tfidf__max_features": range(300, 1000, 100),
                "vect__subject_tfidf__tfidf__max_features": range(30, 110, 10),
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

    classify(
        classifiers,
        args.training_folder,
        args.testing_folder,
        force=args.force,
        plot=args.plot,
    )
