from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec


if __name__ == '__main__':
    w2v_model = Word2Vec(LineSentence('bigram_sentences.txt'))
