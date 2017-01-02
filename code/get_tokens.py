from typing import List
from toolz import compose
from toolz import curry
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import spacy


def tokenizer(tokenizer_type):

    def doc2tokens(doc: str)->List[str]:
        return tokenizer_type.tokenize(doc)
    return doc2tokens

regexptokens = tokenizer(RegexpTokenizer('[\w]+'))


def stemmer(stemmer_type):

    def stem(tokens: List[str])->List[str]:
        return [stemmer_type.stem(token) for token in tokens]
    return stem

snowball = stemmer(SnowballStemmer('english'))


@curry
def remove_stopwords(stopfile: str, tokens: List[str])->List[str]:
    with open(stopfile, 'r', encoding='utf-8') as f:
        stops = set([line.strip() for line in f])
    return [token for token in tokens if token not in stops]


def remove_nonascii(s: str)->str:
        return "".join(i for i in s if ord(i) < 128)


def remove_symbols(tokens: List[str])->List[str]:
        symbols = {' ', '#', '$', '%', '&', "'", '(', ')', '*',
                   '+', ',', '-', '.', '/', ':', '<', '=', '>', '@',
                   '[', '\\', ']', '^', '{', '|', '}'}
        return [token for token in tokens if token not in symbols]


def list2whitespace(tokens: List[str])->str:
    return ' '.join(tokens)


def skl_analyzer(doc: str)->List[str]:
    return compose(
                   # snowball,
                   remove_stopwords('combined-stop-words.txt'),
                   regexptokens,
                   remove_nonascii)(doc)


def tokensfrom_multipledocs(filename: str)->List[List[str]]:
    '''Generate tokens from a file where each
    document is a string terminated by a newline
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        return [skl_analyzer(line.strip()) for line in f]


def make_parser(parser: spacy.en.English):
    def docs_spacy(docstr: str)->spacy.tokens.doc.Doc:
        return parser(docstr)
    return docs_spacy

nlp = make_parser(spacy.load('en'))


def tokens_spacy(doc: spacy.tokens.doc.Doc)->List[str]:
    return [tokens.lemma_ for tokens in doc
            if tokens.pos_ != 'PUNCT']


def spacy_analyzer(docstr: str)->List[str]:
    return compose(remove_symbols,
                   remove_stopwords('combined-stop-words.txt'),
                   tokens_spacy,
                   nlp,
                   remove_nonascii)(docstr)


def multipledocs_spacy(filename: str)->List[List[str]]:
    '''Generate tokens from a file where each
    document is a string terminated by a newline
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        return [spacy_analyzer(line.strip()) for line in f]
