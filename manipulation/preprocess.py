import contractions
import string
import nltk
import unicodedata

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

def normalize(content, word_length_limit=20):
    content = content.replace('\n', ' ').replace('\r', ' ')
    wnl = WordNetLemmatizer()

    words    = nltk.word_tokenize(content)
    # words[:] = filter(lambda w: len(w) <= word_length_limit, words)
    words[:] = map(lambda w: unicodedata.normalize('NFKD',  w), words)
    words[:] = map(lambda w: w.encode('ascii', 'ignore').decode('utf-8', 'ignore'), words)
    words[:] = map(lambda w: w.lower(), words)
    # words[:] = filter(lambda w: not all(c in string.punctuation for c in w), words)
    # words[:] = filter(lambda w: not all(c.isdigit() for c in w), words)
    words[:] = filter(lambda w: w not in nltk.corpus.stopwords.words('english'), words)
    words[:] = map(lambda w: wnl.lemmatize(w, pos=_get_wordnet_pos(w)).strip(), words)
    words[:] = filter(lambda w: w != '', words)

    return ' '.join(words)

def preprocess(file):
    with file.open() as fp:
        denoised = denoise(fp.read())
        normalized = normalize(denoised)
    return normalized

def preprocess_files(input_folder, output_folder, force=False):
    utils.map_files(preprocess, output_folder, input_folder.iterdir(), force=force)

if __name__ == '__main__':
    input_folder, output_folder = utils.io_folder_argparse()
    preprocess_files(input_folder, output_folder)
