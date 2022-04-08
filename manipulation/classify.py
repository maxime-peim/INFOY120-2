# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from sklearn import metrics
from functools import partial

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import density

from . import utils

def benchmark(clf, output_folder, X_train, y_train, X_test, y_test=None):
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
    
    score = 0
    if y_test is not None:
        score = metrics.accuracy_score(y_test, pred)
        
        print("classification report:")
        print(metrics.classification_report(y_test, pred))

        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
    
    else:
        clf_folder = output_folder.joinpath(f'{clf_descr}')
        clf_folder.mkdir(exist_ok=True, parents=True)
        pred_df.applymap(lambda x: int(x != 'spam')) \
            .to_csv(clf_folder.joinpath('labels.csv'), index_label='id')

        with clf_folder.joinpath('params.txt').open('w') as fp:
            fp.write(repr(clf))
    
    return clf_descr, score, train_time, test_time

def classify(train_df, test_df, classifiers, *, training_folder, testing_folder, select_chi2, split_training=None, force=False, plot=False):

    y_train = train_df.prediction
    y_test = None

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    #vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
    X_train = vectorizer.fit_transform(train_df.content)
    duration = time() - t0
    print(f"done in {duration:.03}s")
    print("n_samples: {0[0]}, n_features: {0[1]}\n".format(X_train.shape))
    
    if split_training is not None:
        print("Splitting the train dataset\n")
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=split_training, random_state=42)
    else:
        print("Extracting features from the test data using the same vectorizer")
        t0 = time()
        X_test = vectorizer.transform(test_df.content)
        duration = time() - t0
        print(f"done in {duration:.03}s")
        print("n_samples: {0[0]}, n_features: {0[1]}\n".format(X_test.shape))

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

    output_folder = testing_folder.joinpath("labels")
    output_folder.mkdir(exist_ok=True, parents=True)
    
    results = []
    for clf in classifiers:
        print("=" * 80)
        print(repr(clf))
        results.append(benchmark(clf, output_folder, X_train, y_train, X_test, y_test=y_test))

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