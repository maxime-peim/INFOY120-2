import pathlib
import manipulation

TRAINING_DATA = pathlib.Path('data/TR')
TESTING_DATA = pathlib.Path('data/TT')

def extract_and_preprocess(data_folder):
    raw_folder = data_folder.joinpath('raw')
    extracted_folder = data_folder.joinpath('extracted')
    preprocessed_folder = data_folder.joinpath('preprocessed')
    
    manipulation.extract_files(raw_folder, extracted_folder)
    manipulation.preprocess_files(extracted_folder, preprocessed_folder)

if __name__ == '__main__':
    extract_and_preprocess(TRAINING_DATA)
    extract_and_preprocess(TESTING_DATA)
