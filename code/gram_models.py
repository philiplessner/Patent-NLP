from typing import Iterator, List
import spacy
import gensim
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
import global_constants
from utilities import save2file, save_model


nlp = spacy.load('en')


def doc2sents(filename: str)->str:
    with open(filename, 'r', encoding='utf-8') as f:
        sentences = []  # type: List[str]
        for doc in nlp.pipe((line.strip() for line in f),
                            batch_size=5000, n_threads=-1):
            for sent in doc.sents:
                newsent = ' '.join((token.lemma_
                                    for token in sent
                                    if not token.is_punct))
                newsent += '\n'
                sentences.append(newsent)
    return ''.join(sentences)


def xgram_model(filename: str)->gensim.models.phrases.Phrases:
    return Phraser(Phrases(LineSentence(filename)))


def remove_stopwords(stopfile: str, tokens: Iterator[str])->Iterator[str]:
    with open(stopfile, 'r', encoding='utf-8') as f:
        stops = set([line.strip() for line in f])
    return (token for token in tokens if token not in stops)


def xgram_strings(filename: str,
                  xgram_model: gensim.models.phrases.Phraser)->str:
    return '\n'.join(' '.join(xgram_model[sentences])
                     for sentences in LineSentence(filename))


if __name__ == '__main__':
    sentences = doc2sents(global_constants.SOURCE_FILE)
    save2file(global_constants.UNI_SENTS, sentences)
    bigram_model = xgram_model(global_constants.UNI_SENTS)
    save_model(global_constants.BI_MODEL, bigram_model)
    bigram_sentences = xgram_strings(global_constants.UNI_SENTS,
                                     bigram_model)
    save2file(global_constants.BI_SENTS, bigram_sentences)
    trigram_model = xgram_model(global_constants.BI_SENTS)
    save_model(global_constants.TRI_MODEL, trigram_model)
    trigram_sentences = xgram_strings(global_constants.BI_SENTS, trigram_model)
    save2file(global_constants.TRI_SENTS, trigram_sentences)
