import sys
import pathlib
import argparse

RAW_DIR = 'raw'
EXTRACTED_DIR = 'extracted'
PREPROCESSED_DIR = 'preprocessed'

PREPROCESSED_FILE = 'preprocessed.csv'
LABELS_FILE = 'labels.csv'

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

def map_files(function, output, files):
    output.mkdir(parents=True, exist_ok=True)
    
    for i, file in enumerate(files, start=1):
        sys.stdout.write("\033[K")
        print(f"[{i:4}] {function.__name__.capitalize()}: {file.name}")
        sys.stdout.write("\033[F")

        processed = function(file)
        processed_path = output.joinpath(file.stem)
        with processed_path.open('w') as processed_file:
            processed_file.write(processed)