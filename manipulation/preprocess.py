from __future__ import annotations

import re
from typing import TYPE_CHECKING

import nltk
import pandas as pd
import spacy
from pandarallel import pandarallel

from . import utils

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame

nlp = spacy.load("en_core_web_sm")
ENGLISH_WORDS = nltk.corpus.words.words()
STOP_WORDS = nltk.corpus.stopwords.words("english")


def _normalize_whitespace(text: str) -> str:
    """
    This function normalizes whitespaces, removing duplicates.
    """
    corrected = re.sub(r"//t", r" ", text)
    corrected = re.sub(r"( )+", r" ", corrected)
    corrected = re.sub(r"(\n)+", r" ", corrected)
    corrected = re.sub(r"(\r)+", r" ", corrected)
    corrected = re.sub(r"(\t)+", r" ", corrected)
    return corrected.strip(" ")


def normalize_subject(subject: str) -> str:
    corrected = str(subject).lower()
    corrected = _normalize_whitespace(corrected)

    doc = nlp(corrected)
    tokens = [token.lemma_ for token in doc]
    tokens[:] = map(
        lambda w: w.encode("ascii", "ignore").decode("utf-8", "ignore"), tokens
    )
    tokens[:] = filter(lambda w: w not in STOP_WORDS, tokens)
    corrected = " ".join(tokens)

    return corrected


def normalize_content(content: str) -> str:
    corrected = str(content).lower()
    corrected = _normalize_whitespace(corrected)

    doc = nlp(corrected)
    tokens = [token.lemma_ for token in doc]
    tokens[:] = map(
        lambda w: w.encode("ascii", "ignore").decode("utf-8", "ignore"), tokens
    )
    tokens[:] = filter(lambda w: w.isalpha() and w in ENGLISH_WORDS, tokens)
    tokens[:] = filter(lambda w: w not in STOP_WORDS, tokens)
    tokens[:] = filter(lambda w: 3 < len(w), tokens)
    corrected = " ".join(tokens)

    return corrected


def preprocess(folder: Path, extracted_df=None, force=False) -> DataFrame:
    print("Preprocessing...", flush=True)
    preprocessed_file = folder.joinpath(utils.PREPROCESSED_FILE)

    if preprocessed_file.exists() and not force:
        print(" loaded from file", flush=True)
        preprocessed_df = pd.read_csv(preprocessed_file, index_col="id")
        preprocessed_df.fillna("", inplace=True)
        return preprocessed_df

    pandarallel.initialize(progress_bar=True)
    nltk.download("stopwords")
    nltk.download("words")

    if extracted_df is None:
        extracted_file = folder.joinpath(utils.EXTRACTED_FILE)
        if not extracted_file.exists():
            raise FileNotFoundError("No extracted content.")
        extracted_df = pd.read_csv(extracted_file, index_col="id")

    preprocessed_df = extracted_df.copy()
    normalization_functions = {}
    for attr_name, func in globals().items():
        if attr_name.startswith("normalize_"):
            name = attr_name.split("_")[1]
            normalization_functions[name] = func

    for column in preprocessed_df.columns:
        if column in normalization_functions:
            print(f" preprocessing {column}...", flush=True)
            preprocessed_df[column] = preprocessed_df[column].parallel_apply(
                normalization_functions[column]
            )

    preprocessed_df.fillna("", inplace=True)
    preprocessed_df.to_csv(preprocessed_file)

    print(" done preprocessing", flush=True)
    return preprocessed_df
