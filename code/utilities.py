from itertools import islice
import pickle
from typing import List
from toolz import curry
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.similarities.docsim import MatrixSimilarity


@curry
def save2file(filepath: str, tosave: str)->str:
    '''Save a string to a file.
    Parameters
        filepath: full path to file relative to calling program
        tosave: string to save to file
        filepath: passed through for use in subsequent procedures
    '''
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(tosave)
    return filepath


def read_file(filepath: str)->str:
    '''Read a utf-8 encoded text file and place results in a string
    Parameters
        filepath: full string path to file
    Returns
        contents: a string contaning contents of file
    '''
    with open(filepath, 'r', encoding='utf-8') as f:
        contents = f.read()
    return contents


def lines_from_file(filepath: str, line: int)->str:
    '''Read a specific line from a file.
    Parameters
        filepath: full string path to file to read from
        line: line in file to read (lines start at 0)
    Returns
        line as string
    '''
    with open(filepath, 'r', encoding='utf-8') as f:
        contents = list(islice(f, line, line + 1))
    return contents[0]


@curry
def save_model(filepath: str, model):
    model.save(filepath)


def load_xgram_model(filepath: str)->gensim.models.phrases.Phrases:
    return Phrases.load(filepath)


def load_w2v_model(filepath: str)->gensim.models.word2vec.Word2Vec:
    return Word2Vec.load(filepath)


def load_dictionary(filepath: str)->gensim.corpora.dictionary.Dictionary:
    return Dictionary.load(filepath)


def load_corpus(filepath: str)->gensim.corpora.mmcorpus.MmCorpus:
    return MmCorpus(filepath)


def load_index(filepath: str)->gensim.similarities.docsim.MatrixSimilarity:
    return MatrixSimilarity.load(filepath)


def serialize_tokens(filename: str, tokens: List[List[str]])->None:
    with open(filename, 'wb') as f:
        pickle.dump(tokens, f)
