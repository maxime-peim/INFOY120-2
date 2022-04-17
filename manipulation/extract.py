from __future__ import annotations

import email
import sys
from typing import TYPE_CHECKING

import pandas as pd

from . import utils

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame

    EmailData = dict[str, str]


def extract_from_file(eml_file: Path) -> EmailData:
    """Extract meaningful data from email.

    Args:
        eml_file (pathlib.Path): pathlib object to an email file.

    Returns:
        dict[str, str]: dictionary from email data name to actual data.
    """
    if not eml_file.exists():
        sys.exit(f"ERROR: input file does not exist: {eml_file}")

    with eml_file.open("r", encoding="latin-1") as fp:
        parsed = email.message_from_file(fp)

    subject = parsed.get("subject", "")
    from_ = parsed.get("from", "")
    payload = ""
    for part in parsed.walk():
        if part.get_content_type() == "text/html":
            payload += utils.denoise_html(part.get_payload())
        elif part.get_content_type() == "text/plain":
            payload += part.get_payload()

    return {"from": from_, "subject": subject, "content": payload}


def extract(folder: Path, force=False) -> DataFrame:
    print("Extracting... ", flush=True)
    raw_folder = folder.joinpath(utils.RAW_DIR)
    labels_file = folder.joinpath(utils.LABELS_FILE)
    extracted_file = folder.joinpath(utils.EXTRACTED_FILE)

    if extracted_file.exists() and not force:
        print(" loaded from file", flush=True)
        extracted_df = pd.read_csv(extracted_file, index_col="id")
        extracted_df.fillna("", inplace=True)
        return extracted_df

    files = raw_folder.glob("*.eml")
    data = {"id": [], "from": [], "subject": [], "content": [], "content_length": []}
    for i, file in enumerate(files, start=1):
        sys.stdout.write("\033[K")
        print(f"[{i:4}] Extraction: {file.name}")
        sys.stdout.write("\033[F")

        id = int(file.stem.split("_")[1])
        email_data = extract_from_file(file)

        data["id"].append(id)
        for data_name, data_value in email_data.items():
            data[data_name].append(data_value)
        data["content_length"].append(len(email_data.get("content", "")))

    extracted_df = pd.DataFrame.from_dict(data)
    extracted_df.set_index("id", inplace=True)

    if labels_file.exists():
        labels_df = pd.read_csv(labels_file, index_col="id")
    else:
        labels_df = pd.DataFrame([""] * extracted_df.shape[0], columns=["prediction"])
        labels_df.index.name = "id"
        labels_df.index += 1

    extracted_df = pd.concat((extracted_df, labels_df), axis="columns")
    extracted_df.fillna("", inplace=True)
    extracted_df.to_csv(extracted_file)

    sys.stdout.write("\033[K")
    print(" done extracting", flush=True)
    return extracted_df
