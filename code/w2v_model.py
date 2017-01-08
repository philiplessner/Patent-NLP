from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
import global_constants
from utilities import save_model


if __name__ == '__main__':
    filepath = '../intermediate/trigram_sentences_cleaned.txt'
    w2v_model = Word2Vec(LineSentence(filepath))
    save_model('../models/w2v.model', w2v_model)
