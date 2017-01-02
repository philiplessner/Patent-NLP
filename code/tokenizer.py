import pickle
from typing import List, Iterator
from toolz import compose
from toolz import curry
import spacy
from gensim.models.phrases import Phrases
import global_constants


@curry
def remove_stopwords(stopfile: str, tokens: Iterator[str])->Iterator[str]:
    with open(stopfile, 'r', encoding='utf-8') as f:
        stops = set([line.strip() for line in f])
    return (token for token in tokens if token not in stops)


def remove_nonascii(s: str)->str:
        return "".join(i for i in s if ord(i) < 128)


def remove_symbols(tokens: Iterator[str])->Iterator[str]:
        symbols = {' ', '#', '$', '%', '&', "'", '(', ')', '*',
                   '+', ',', '-', '.', '/', ':', '<', '=', '>', '@',
                   '[', '\\', ']', '^', '{', '|', '}'}
        return (token for token in tokens if token not in symbols)


def make_parser(parser: spacy.en.English):
    def docs_spacy(docstr: str)->spacy.tokens.doc.Doc:
        return parser(docstr)
    return docs_spacy

nlp = make_parser(spacy.load('en'))


def tokens_spacy(doc: spacy.tokens.doc.Doc)->Iterator[str]:
    return (tokens.lemma_ for tokens in doc
            if tokens.pos_ != 'PUNCT')


def xgrams(tokens: List[List[str]])->List[List[str]]:
    xgram_model = Phrases(tokens)
    return [xgram_model[token] for token in tokens]


def analyzer_spacy(docstr: str)->List[str]:
    return compose(list,
                   remove_symbols,
                   remove_stopwords('combined-stop-words.txt'),
                   tokens_spacy,
                   nlp,
                   remove_nonascii)(docstr)


def file_line_gen(filename: str)->Iterator[str]:
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        yield line


def multipledocs_spacy(filename: str)->List[List[str]]:
    '''Generate tokens from a file where each
    document is a string terminated by a newline
    '''
    flg = file_line_gen(filename)
    return [analyzer_spacy(line.strip())for line in flg]


def uni2xgrams(tokens: List[List[str]])->List[List[str]]:
        return compose(xgrams,
                       xgrams)(tokens)


def serialize_tokens(filename: str, tokens: List[List[str]])->None:
    with open(filename, 'wb') as f:
        pickle.dump(tokens, f)


if __name__ == '__main__':
    bi_tokens = multipledocs_spacy('abstractclaims_bigrams.txt')
    # tri_tokens = uni2xgrams(uni_tokens)
    serialize_tokens(global_constants.TOKENS_FILE, bi_tokens)
