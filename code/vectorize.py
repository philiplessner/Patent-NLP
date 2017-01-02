import pickle
from typing import List
from gensim import corpora, models
import global_constants


def unpickle(filename: str)->List[List[str]]:
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    tokens = unpickle(global_constants.TOKENS_FILE)
    dictionary = corpora.Dictionary(tokens)
    patent_corpus = [dictionary.doc2bow(token) for token in tokens]
    tfidf = models.tfidfmodel.TfidfModel(corpus=patent_corpus,
                                         id2word=dictionary)
    patent_tfidf = [tfidf[c] for c in patent_corpus]
    dictionary.save(global_constants.DICT_FILE)
    corpora.MmCorpus.serialize(global_constants.COUNTVEC_FILE,
                               patent_corpus,
                               id2word=dictionary)
    corpora.MmCorpus.serialize(global_constants.TFDIFVEC_FILE,
                               patent_tfidf,
                               id2word=dictionary)
