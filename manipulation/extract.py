import sys
import email
import pandas as pd

from . import utils

def extract_from_file(eml_file):
    if not eml_file.exists():
        sys.exit(f"ERROR: input file does not exist: {eml_file}")
    
    with eml_file.open('rb') as fp:
        parsed = email.message_from_binary_file(fp)

    subject = str(parsed.get('subject', ''))
    from_ = str(parsed.get('from', ''))
    payload = parsed.get_payload()
    if isinstance(payload, list):
        payload = payload[0]
    if not isinstance(payload, str):
        payload = str(payload)
    
    extracted = [from_, subject, payload]
    return '\n'.join(extracted)

def extract(folder, force=False):
    print("Extracting... ", flush=True)
    raw_folder = folder.joinpath(utils.RAW_DIR)
    labels_file = folder.joinpath(utils.LABELS_FILE)
    extracted_file = folder.joinpath(utils.EXTRACTED_FILE)
    
    if extracted_file.exists() and not force:
        print(" loaded from file", flush=True)
        extracted_df = pd.read_csv(extracted_file, index_col='id')
        extracted_df.fillna('', inplace=True)
        return extracted_df
    
    files = raw_folder.glob("*.eml")
    data = {'id': [], 'content': [], 'length': []}
    for i, file in enumerate(files, start=1):
        sys.stdout.write("\033[K")
        print(f"[{i:4}] Extraction: {file.name}")
        sys.stdout.write("\033[F")
        
        id = int(file.stem.split("_")[1])
        extracted = extract_from_file(file)
        
        data['id'].append(id)
        data['content'].append(extracted)
        data['length'].append(len(extracted))
    
    extracted_df = pd.DataFrame.from_dict(data)
    extracted_df.set_index('id', inplace=True)
    
    if labels_file.exists():
        labels_df = pd.read_csv(labels_file, index_col='id')
    else:
        labels_df = pd.DataFrame(['no_prediction']*extracted_df.shape[0], columns=['prediction'])
        labels_df.index.name = 'id'
        labels_df.index += 1
    
    extracted_df = pd.concat((extracted_df, labels_df), axis='columns')
    extracted_df.fillna('', inplace=True)
    print(extracted_df)
    extracted_df.to_csv(extracted_file)
    
    print(" done extracting", flush=True)
    return extracted_df
