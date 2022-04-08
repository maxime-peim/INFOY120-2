import pathlib
import argparse
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

import manipulation
import manipulation.utils as utils

TRAINING_DATA = pathlib.Path('data/TR')
TESTING_DATA = pathlib.Path('data/TT')

def extract_and_preprocess(data_folder, force_extract=False, force_preprocess=False):
    extracted_df = manipulation.extract(data_folder, force=force_extract)
    processed_df = manipulation.preprocess(data_folder, extracted_df, force=force_preprocess)
    return processed_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-folder",
        type=pathlib.Path,
        default=TRAINING_DATA
    )
    parser.add_argument(
        "--testing-folder",
        type=pathlib.Path,
        default=TESTING_DATA
    )
    parser.add_argument(
        "-f",
        "--force",
        nargs='*',
        choices=['extract', 'preprocess']
    )
    parser.add_argument(
        "--preprocess-file",
        type=str,
        default="preprocessed.csv"
    )
    parser.add_argument(
        "--extracted-file",
        type=str,
        default="preprocessed.csv"
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true"
    )

    parser.add_argument(
        "--chi2-select",
        type=int,
        dest="select_chi2",
        help="Select some number of features using a chi-squared test",
    )
    parser.add_argument(
        "-s",
        "--split",
        "--split-training",
        dest="split_training",
        type=int,
        default=None
    )

    args = parser.parse_args()
    force_extract = 'extract' in args.force if args.force is not None else False
    force_preprocess = ('preprocess' in args.force if args.force is not None else False) or force_extract

    train_df = extract_and_preprocess(TRAINING_DATA, force_extract=force_extract, force_preprocess=force_preprocess)
    test_df = extract_and_preprocess(TESTING_DATA, force_extract=force_extract, force_preprocess=force_preprocess)
    
    classify_kwargs = vars(args)
    classify_kwargs['force'] = classify_kwargs['force'] is not None
    
    classifiers = [
        # RidgeClassifier(tol=1e-2, solver="sag"),
        # Perceptron(max_iter=50),
        # PassiveAggressiveClassifier(max_iter=50),
        # KNeighborsClassifier(n_neighbors=10),
        # RandomForestClassifier(),
        # LinearSVC(penalty='l1', dual=False, tol=1e-3),
        # LinearSVC(penalty='l2', dual=False, tol=1e-3),
        # SGDClassifier(alpha=0.0001, max_iter=50, penalty='l1'),
        # SGDClassifier(alpha=0.0001, max_iter=50, penalty='l2'),
        # SGDClassifier(alpha=0.0001, max_iter=50, penalty="elasticnet"),
        # NearestCentroid(),
        # GaussianNB()
        MultinomialNB(),
        # BernoulliNB(alpha=0.01),
        # ComplementNB(alpha=0.1),
        # LogisticRegression(**{'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}),
        # Pipeline([("feature_selection", SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3)),), ("classification", LinearSVC(penalty="l2")),])
    ]
    
    manipulation.classify(train_df, test_df, classifiers, **classify_kwargs)
