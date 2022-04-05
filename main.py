import pathlib
import argparse

import manipulation
import manipulation.utils as utils

TRAINING_DATA = pathlib.Path('data/TR')
TESTING_DATA = pathlib.Path('data/TT')

def extract_and_preprocess(data_folder, force_extract=False, force_preprocess=False):
    raw_folder = data_folder.joinpath(utils.RAW_DIR)
    extracted_folder = data_folder.joinpath(utils.EXTRACTED_DIR)
    preprocessed_folder = data_folder.joinpath(utils.PREPROCESSED_DIR)
    
    manipulation.extract_files(raw_folder, extracted_folder, force=force_extract)
    manipulation.preprocess_files(extracted_folder, preprocessed_folder, force=force_preprocess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--force",
        nargs='*',
        choices=['extract', 'preprocess']
    )

    parser.add_argument(
        "--chi2_select",
        type=int,
        dest="select_chi2",
        help="Select some number of features using a chi-squared test",
    )
    parser.add_argument(
        "--top10",
        action="store_true",
        dest="print_top10",
        help="Print ten most discriminative terms per class for every classifier.",
    )
    parser.add_argument("--use_hashing", action="store_true", help="Use a hashing vectorizer.")
    parser.add_argument(
        "--n_features",
        type=int,
        default=2 ** 16,
        help="n_features when using the hashing vectorizer.",
    )

    args = parser.parse_args()
    force_extract = 'extract' in args.force if args.force is not None else False
    force_preprocess = 'preprocess' in args.force if args.force is not None else False

    extract_and_preprocess(TRAINING_DATA, force_extract=force_extract, force_preprocess=force_preprocess)
    extract_and_preprocess(TESTING_DATA, force_extract=force_extract, force_preprocess=force_preprocess)
    
    classify_kwargs = vars(args)
    classify_kwargs['force'] = classify_kwargs['force'] is not None
    
    manipulation.classify(TRAINING_DATA, TESTING_DATA, **classify_kwargs)
