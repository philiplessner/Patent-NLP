from typing import List, Tuple, Union
import gensim
from gensim import corpora, models, similarities
import global_constants


Vec = List[List[Tuple[int, Union[int, float]]]]


def model_lda(vectors: Vec,
              dictionary: gensim.corpora.dictionary.Dictionary,
              num_topics=100):
    return models.LdaModel(vectors,
                           id2word=dictionary,
                           num_topics=num_topics)

if __name__ == '__main__':
    patent_tfidf = corpora.MmCorpus(global_constants.TFDIFVEC_FILE)
    dictionary = corpora.Dictionary.load(global_constants.DICT_FILE)
    lda = model_lda(patent_tfidf, dictionary)
    index = similarities.MatrixSimilarity(lda[patent_tfidf])
    lda.save(global_constants.LDAMODEL_FILE)
    index.save(global_constants.INDEX_FILE)
