from typing import List, Tuple
from toolz.sandbox.core import unzip
from cytoolz import compose, curry
from cytoolz.curried import get
from cytoolz.curried import map as cmap
import numpy as np
import pandas as pd
import gensim
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
import global_constants
from utilities import save_model, load_w2v_model


Vocab = List[Tuple[str, int, int]]
Vocab_Transposed = Tuple[List[str], List[int], List[int]]


def model2vocab(model: gensim.models.word2vec.Word2Vec) -> Vocab:
    return [(term, voc.index, voc.count)
            for term, voc in model.vocab.items()]


def sort_vocab(vocabulary: Vocab) -> Vocab:
    return sorted(vocabulary, key=lambda x: -x[2])


@curry
def words2matrix(model: gensim.models.word2vec.Word2Vec,
                 ordered_terms: List[str]) -> np.ndarray:
    return np.array([model[word] for word in ordered_terms])


def vocab_items(model: gensim.models.word2vec.Word2Vec) -> Vocab_Transposed:
    return compose(list,
                   cmap(list),
                   unzip,
                   sort_vocab,
                   model2vocab)(model)


model2matrix = compose(words2matrix(load_w2v_model(global_constants.W2V_MODEL)),
                       get(0),
                       vocab_items)


def matrix2dataframe(index: List[str], matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=index)


if __name__ == '__main__':
    filepath = '../intermediate/trigram_sentences_cleaned.txt'
    w2v_model = Word2Vec(LineSentence(filepath),
                         size=300,
                         sg=1,
                         sample=1.e-5)
    save_model(global_constants.W2V_MODEL, w2v_model)
