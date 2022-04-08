import contractions
import re
import nltk
import unicodedata
import pandas as pd

from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from . import utils

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

def denoise(content):
    content = BeautifulSoup(content, 'html.parser').get_text()
    content = contractions.fix(content)
    return content

def _get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def _replace_urls(text):
    url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    text = re.sub(url_regex, "URL", text)
    return text

def _replace_money(text):
    money_regex = r'(NUMBER\s?(\$| dollars?))'
    text = re.sub(money_regex, "MONEY", text)
    return text

def _simplify_punctuation(text):
    """
    This function simplifies doubled or more complex punctuation. The exception is '...'.
    """
    corrected = str(text)
    corrected = re.sub(r'([!?,;])+', r'', corrected)
    corrected = re.sub(r'\.{2,}', r'...', corrected)
    return corrected

def _simplify_number(text):
    """
    This function simplifies doubled or more complex punctuation. The exception is '...'.
    """
    corrected = str(text)
    corrected = re.sub(r'((\d+\.?\d*)|(\d*\.?\d+))', r'NUMBER', corrected)
    return corrected

def _normalize_whitespace(text):
    """
    This function normalizes whitespaces, removing duplicates.
    """
    corrected = re.sub(r"//t",r" ", text)
    corrected = re.sub(r"( )+",r" ", corrected)
    corrected = re.sub(r"(\n)+",r" ", corrected)
    corrected = re.sub(r"(\r)+",r" ", corrected)
    corrected = re.sub(r"(\t)+",r" ", corrected)
    return corrected.strip(" ")

def _simplify(text):
    corrected = text.lower()
    corrected = _normalize_whitespace(corrected)
    corrected = _simplify_punctuation(corrected)
    corrected = _replace_urls(corrected)
    corrected = _simplify_number(corrected)
    corrected = _replace_money(corrected)
    return corrected

def normalize(content):
    content = _simplify(content)
    wnl = WordNetLemmatizer()

    words    = nltk.word_tokenize(content)
    # words[:] = filter(lambda w: len(w) <= word_length_limit, words)
    words[:] = filter(lambda w: w.isalpha(), words)
    words[:] = map(lambda w: unicodedata.normalize('NFKD',  w), words)
    words[:] = map(lambda w: w.encode('ascii', 'ignore').decode('utf-8', 'ignore'), words)
    # words[:] = filter(lambda w: not all(c in string.punctuation for c in w), words)
    # words[:] = filter(lambda w: not all(c.isdigit() for c in w), words)
    words[:] = filter(lambda w: w not in nltk.corpus.stopwords.words('english'), words)
    words[:] = map(lambda w: wnl.lemmatize(w, pos=_get_wordnet_pos(w)).strip(), words)
    words[:] = filter(lambda w: w != '', words)

    return ' '.join(words)

def denoise_and_normalize(content):
    return normalize(denoise(content))

def preprocess(folder, extracted_df=None, force=False):
    print("Preprocessing... ", end="", flush=True)
    preprocessed_file = folder.joinpath(utils.PREPROCESSED_FILE)
    
    if preprocessed_file.exists() and not force:
        print("loaded from file", flush=True)
        preprocessed_df = pd.read_csv(preprocessed_file, index_col='id')
        preprocessed_df.fillna('', inplace=True)
        return preprocessed_df
    
    if extracted_df is None:
        extracted_file = folder.joinpath(utils.EXTRACTED_FILE)
        if not extracted_file.exists():
            raise FileNotFoundError("No extracted content.")
        extracted_df = pd.read_csv(extracted_file, index_col='id')
    
    preprocessed_df = extracted_df.copy()
    preprocessed_df['content'] = preprocessed_df['content'].apply(denoise_and_normalize)
    preprocessed_df.fillna('', inplace=True)
    preprocessed_df.to_csv(preprocessed_file)
    
    print("done preprocessing", flush=True)
    return preprocessed_df
