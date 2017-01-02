from itertools import islice
import pickle
from typing import List
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.similarities.docsim import MatrixSimilarity


def save2file(filepath: str, tosave: str)->None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(tosave)


def read_file(filepath: str)->str:
    with open(filepath, 'r', encoding='utf-8') as f:
        contents = f.read()
    return contents


def lines_from_file(filepath: str, line: int)->str:
    with open(filepath, 'r', encoding='utf-8') as f:
        contents = list(islice(f, line, line + 1))
    return contents[0]


def save_model(filepath: str, model)->None:
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
