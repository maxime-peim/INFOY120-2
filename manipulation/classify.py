# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from . import utils

def load_preprocessed_data(data_folder, force=False):
    if isinstance(data_folder, str):
        data_folder = pathlib.Path(data_folder)
    
    save = True
    if data_folder.joinpath(utils.PREPROCESSED_FILE).exists() and not force:
        save = False
        data_df = pd.read_csv(data_folder.joinpath(utils.PREPROCESSED_FILE))
        data_df.set_index('id', inplace=True)
    elif data_folder.joinpath(utils.LABELS_FILE).exists():
        data_df = pd.read_csv(data_folder.joinpath(utils.LABELS_FILE))
        data_df.set_index("id", inplace=True)
        
        training_folder = data_folder.joinpath(utils.PREPROCESSED_DIR)
        data = {}
        for file in training_folder.iterdir():
            id = int(file.stem.split('_')[1])
            with file.open() as fp:
                data[id] = fp.read()
        mails = pd.DataFrame.from_dict(data, orient="index", columns=["data"])
        data_df = pd.concat((data_df, mails), axis='columns')
    else:
        training_folder = data_folder.joinpath(utils.PREPROCESSED_DIR)
        data = {}
        for file in training_folder.iterdir():
            id = int(file.stem.split('_')[1])
            with file.open() as fp:
                data[id] = fp.read()
        data_df = pd.DataFrame.from_dict(data, orient="index", columns=["data"])
        data_df.index.name = 'id'
        data_df['prediction'] = 'no_prediction'
    
    if save:
        data_df.to_csv(data_folder.joinpath(utils.PREPROCESSED_FILE), index_label='id')
    return data_df


def classify(training_folder, testing_folder, *, select_chi2, print_top10, use_hashing, n_features, force=False):
    
    print("Loading preprocessed data... ", end="", flush=True)
    
    data_train = load_preprocessed_data(training_folder, force=force)
    data_test = load_preprocessed_data(testing_folder, force=force)
    y_train = data_train.prediction
    
    print("data loaded", flush=True)

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    if use_hashing:
        vectorizer = HashingVectorizer(
            stop_words="english", alternate_sign=False, n_features=n_features
        )
        X_train = vectorizer.transform(data_train.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")
        X_train = vectorizer.fit_transform(data_train.data)
    duration = time() - t0
    print(f"done in {duration:.03}s")
    print("n_samples: {0[0]}, n_features: {0[1]}\n".format(X_train.shape))

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    duration = time() - t0
    print(f"done in {duration:.03}s")
    print("n_samples: {0[0]}, n_features: {0[1]}\n".format(X_test.shape))

    # mapping from integer feature name to original token string
    if use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names_out()

    if select_chi2:
        print("Extracting %d best features by a chi-squared test" % select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names is not None:
            # keep selected feature names
            feature_names = feature_names[ch2.get_support()]
        duration = time() - t0
        print(f"done in {duration:.03}s\n")

    output_folder = pathlib.Path("../data/TT/labels")
    
    def benchmark(clf):
        print("_" * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print(f"train time: {train_time:.03}s")
        
        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print(f"test time: {test_time:.03}s")
        
        clf_descr = repr(clf)
        
        pred_df = pd.DataFrame(pred, columns=["prediction"])
        pred_df.index.name = 'id'
        pred_df.index += 1
        
        score = clf.score(X_train, y_train)
        
        pred_train = clf.predict(X_train)
        print(metrics.confusion_matrix(y_train, pred_train))
        
        if output_folder is not None:
            clf_folder = output_folder.joinpath(f'{clf_descr}')
            clf_folder.mkdir(exist_ok=True, parents=True)
            pred_df.applymap(lambda x: int(x != 'spam')) \
                .to_csv(clf_folder.joinpath('labels.csv'), index_label='id')
        
            with clf_folder.joinpath('params.txt').open('w') as fp:
                fp.write(repr(clf))

        if hasattr(clf, "coef_"):
            print(f"dimensionality: {clf.coef_.shape[1]}")
            print(f"density: {density(clf.coef_)}")

            if print_top10 and feature_names is not None:
                print("top 10 keywords for prediction:")
                top10 = np.argsort(clf.coef_[0])[-10:]
                print(f"prediction: {feature_names[top10]}")
        
        print()
        
        return clf_descr, score, train_time, test_time
    
    results = []
    for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest"),
    ):
        print("=" * 80)
        print(name)
        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        print("=" * 80)
        print(f"{penalty.upper()} penalty")
        # Train Liblinear model
        results.append(benchmark(LinearSVC(penalty=penalty, dual=False, tol=1e-3)))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=0.0001, max_iter=50, penalty=penalty)))

    # Train SGD with Elastic Net penalty
    print("=" * 80)
    print("Elastic-Net penalty")
    results.append(
        benchmark(SGDClassifier(alpha=0.0001, max_iter=50, penalty="elasticnet"))
    )

    # Train NearestCentroid without threshold
    print("=" * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid()))

    # Train sparse Naive Bayes classifiers
    print("=" * 80)
    print("Naive Bayes")
    # results.append(benchmark(GaussianNB(), print_top10=print_top10))
    results.append(benchmark(MultinomialNB(alpha=0.01)))
    results.append(benchmark(BernoulliNB(alpha=0.01)))
    results.append(benchmark(ComplementNB(alpha=0.1)))

    print("=" * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(
        benchmark(
            Pipeline(
                [
                    (
                        "feature_selection",
                        SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3)),
                    ),
                    ("classification", LinearSVC(penalty="l2")),
                ]
            )
        )
    )

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)
    
    best_clf_i = np.argmax(score)
    print(f"Best classifiers: {clf_names[best_clf_i]} -> {score[best_clf_i]}")

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