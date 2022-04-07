import sys
import pathlib
import argparse
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

def io_folder_argparse():
    parser = argparse.ArgumentParser(description='Extract infos from eml files.')
    parser.add_argument('input_folder', metavar='IN', type=pathlib.Path)
    parser.add_argument('output_folder', metavar='OUT', type=pathlib.Path)

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    if not input_folder.exists() or not input_folder.is_dir():
        sys.exit('Input folder must exist.')

    output_folder.mkdir(parents=True, exist_ok=True)
    
    return input_folder, output_folder

def map_files(function, output, files, force=False):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not force:
        return
    
    for i, file in enumerate(files, start=1):
        sys.stdout.write("\033[K")
        print(f"[{i:4}] {function.__name__.capitalize()}: {file.name}")
        sys.stdout.write("\033[F")

        processed = function(file)
        processed_path = output.joinpath(file.stem)
        with processed_path.open('w') as processed_file:
            processed_file.write(processed)
            

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