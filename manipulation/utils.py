from numpy import mean
from sklearn.metrics import (
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate

RAW_DIR = 'raw'
EXTRACTED_DIR = 'extracted'
PREPROCESSED_DIR = 'preprocessed'

EXTRACTED_FILE = 'extracted.csv'
PREPROCESSED_FILE = 'preprocessed.csv'
LABELS_FILE = 'labels.csv'

SCORING = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, pos_label="ham"),
    "Recall": make_scorer(recall_score, pos_label="ham"),
    "F-M": make_scorer(f1_score, pos_label="ham"),
    "MCC": make_scorer(matthews_corrcoef),
    "AUC": "roc_auc",
}

def extract_scores(scores):
    extracted = {}
    for score_name, values in scores.items():
        if score_name.startswith("test_"):
            extracted[score_name[5:]] = mean(values)
    return extracted            

def get_scores(clf, X, y):
    full_scores = cross_validate(
        clf,
        X,
        y,
        scoring=SCORING,
        cv=10,
        n_jobs=-1,
        error_score="raise",
    )
    return extract_scores(full_scores)
