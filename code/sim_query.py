import linecache
from gensim import corpora, models, similarities


def load_dictionary(filepath):
    return corpora.Dictionary.load(filepath)


def load_corpus(filepath):
    return corpora.MmCorpus(filepath)


def load_index(filepath):
    return similarities.MatrixSimilarity.load(filepath)


def query(model, transform, index, qdoc):
    vec_bow = dictionary.doc2bow(qdoc.lower().split())
    vec_transformed = transform[vec_bow]
    vec_model = model[vec_transformed]
    sims = index[vec_model]
    return sorted(enumerate(sims), key=lambda item: -item[1])


def print_query_result(filename, qdoc, qresult, ndocs=10):
    print('Query: ', qdoc, '\n')
    topnresults = qresult[0:ndocs]
    for result in topnresults:
        print('Document Number: ', result[0], '\t', 'Relevance: ', result[1])
        text = linecache.getline(filename, result[0])
        print(text[0:1000])
        print('\n')


if __name__ == '__main__':
    dictionary = load_dictionary('patent.dict')
    patent_corpus = load_corpus('patent.mm')
    tfidf = models.tfidfmodel.TfidfModel(corpus=patent_corpus,
                                         id2word=dictionary)
    index = load_index('patent.index')
    model = models.LdaModel.load('patent_lda.model')
    qdoc = 'electronic circuit capacitor component'
    qresult = query(model, tfidf, index, qdoc)
    print_query_result('abstractclaims.txt', qdoc, qresult)
