import numpy as np
from scipy.spatial import distance
from gensim import corpora, models


def doc_topics(vectors, model):
    return [model[v] for v in vectors]


def topics2array(topics):
    dense = np.zeros((len(topics), 100), float)
    for ti, t in enumerate(topics):
        for tj, v in t:
            dense[ti, tj] = v
    return dense


def pairwise_distance(topics, dense):
    pairwise = distance.squareform(distance.pdist(dense))
    largest = pairwise.max()
    for ti in range(len(topics)):
        pairwise[ti, ti] = largest + 1
    return pairwise


def closest_to(pairwise, doc_id):
    return pairwise[doc_id].argmin()

if __name__ == '__main__':
    patent_tfidf = corpora.MmCorpus('patent_tfidf.mm')
    model = models.LdaModel.load('patent_lda.model')
    topics = doc_topics(patent_tfidf, model)
    dense = topics2array(topics)
    pairwise = pairwise_distance(topics, dense)
