import sys
import email

from . import utils

def extract(eml_file):
    if not eml_file.exists():
        sys.exit(f"ERROR: input file does not exist: {eml_file}")
    
    with eml_file.open('rb') as fp:
        parsed = email.message_from_binary_file(fp)

    sub = str(parsed.get('subject', ''))
    payload = parsed.get_payload()
    if isinstance(payload, list):
        payload = payload[0]
    if not isinstance(payload, str):
        payload = str(payload)
    
    extracted = [sub, payload]
    return '\n'.join(extracted)

def extract_files(input_folder, output_folder):
    utils.map_files(extract, output_folder, input_folder.glob('*.eml'))

if __name__ == '__main__':
    input_folder, output_folder = utils.io_folder_argparse()
    extract_files(input_folder, output_folder)
