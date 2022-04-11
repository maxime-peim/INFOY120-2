import argparse
from pathlib import Path
from random import random

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import manipulation.utils as utils
from manipulation import classify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-folder",
        type=Path,
        default=utils.TRAINING_DATA
    )
    parser.add_argument(
        "--testing-folder",
        type=Path,
        default=utils.TESTING_DATA
    )
    parser.add_argument(
        "-f",
        "--force",
        nargs='*',
        choices=['extract', 'preprocess']
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true"
    )

    args = parser.parse_args()
    
    """
    classifiers = [
        (RandomForestClassifier(random_state=42), {
            #'clf__bootstrap': [True, False], 
            'clf__max_depth': [40],#range(10, 110, 10),
            #'clf__max_features': ['auto', 'sqrt'], 
            #'clf__min_samples_leaf': [1, 2, 4], 
            #'clf__min_samples_split': [2, 5, 10], 
            'clf__n_estimators': range(10, 110, 10),
            'vect__content_tfidf__tfidf__max_features': [900],# range(100, 1000, 100),
            'vect__subject_tfidf__tfidf__max_features': range(1, 11, 1),
        }),
    ]
    """
    classifiers = [
        (DecisionTreeClassifier(), {
            'clf__max_depth': range(10, 110, 10),
            'vect__content_tfidf__tfidf__max_features': range(100, 1000, 100),
            'vect__subject_tfidf__tfidf__max_features': range(10, 110, 10),
        })
    ]
    
    classify(classifiers, args.training_folder, args.testing_folder, force=args.force, plot=args.plot)
