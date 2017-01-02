from typing import List
from gensim import corpora, models, similarities
from get_tokens import multipledocs_spacy


def generate_tokens(filepath: str)->List[List[str]]:
    return multipledocs_spacy(filepath)

if __name__ == '__main__':
    tokens = generate_tokens('abstractclaims.txt')
    # When the dictionary.toke2id is called returns {token: id...}
    dictionary = corpora.Dictionary(tokens)
# list(patent_corpus) is a [[()..]...] where each inner list corresponds
# to a document and each tuple is an (id, token-count) pair
    patent_corpus = [dictionary.doc2bow(token) for token in tokens]
    patent_tfidf = models.tfidfmodel.TfidfModel(corpus=patent_corpus,
                                                id2word=dictionary)
    model = models.LdaModel(patent_tfidf[patent_corpus],
                            id2word=dictionary,
                            num_topics=100)
    index = similarities.MatrixSimilarity(model[patent_corpus])
    dictionary.save('patent.dict')
    corpora.MmCorpus.serialize('patent.mm', patent_corpus)
    patent_tfidf.save('patent.tfidf')
    model.save('patent_lda.model')
    index.save('patent.index')
