from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup

from .extract import extract
from .preprocess import preprocess

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame

RAW_DIR = 'raw'

EXTRACTED_FILE = 'extracted.csv'
PREPROCESSED_FILE = 'preprocessed.csv'
LABELS_FILE = 'labels.csv'

TRAINING_DATA = Path('data/TR')
TESTING_DATA = Path('data/TT')

def denoise_html(content: str) -> str:
    content = BeautifulSoup(content, 'html.parser').get_text()
    return content

def extract_and_preprocess(data_folder: Path, force_extract=False, force_preprocess=False) -> DataFrame:
    """Extract and preprocess emails.
    If already done in the past and force = false, the data are read from files stored on disk.

    Args:
        data_folder (pathlib.Path): folder containing a raw folder with all emails.
        force_extract (bool, optional): Set to true to redo the extraction if already done. Defaults to False.
        force_preprocess (bool, optional): Set to true to redo the preprocessing if already done. Defaults to False.

    Returns:
        pd.DataFrame: dataframe containing preprocessed emails and features.
    """
    extracted_df = extract(data_folder, force=force_extract)
    processed_df = preprocess(data_folder, extracted_df, force=force_preprocess)
    return processed_df