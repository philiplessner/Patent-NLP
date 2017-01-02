from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import Doc2Vec


if __name__ == '__main__':
    d2v_model = Doc2Vec(TaggedLineDocument('abstractclaims_bigrams.txt'))
