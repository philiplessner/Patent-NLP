from typing import Iterator
import spacy
import gensim
from gensim.models.phrases import Phrases
from gensim.models.word2vec import LineSentence
import global_constants
from utilities import save2file, save_model


nlp = spacy.load('en')


def doc2sents(filename: str)->str:
    with open(filename, 'r', encoding='utf-8') as f:
        sentences = ''
        for doc in nlp.pipe((line.strip() for line in f),
                            batch_size=2000, n_threads=3):
            for sent in doc.sents:
                toks = (token.lemma_ for token in sent if not token.is_punct)
                wostops = remove_stopwords('combined-stop-words.txt', toks)
                newsent = ' '.join(list(wostops))
                newsent += '\n'
                sentences += newsent
    return sentences


def xgram_model(filename: str)->gensim.models.phrases.Phrases:
    return Phrases(LineSentence(filename))


def remove_stopwords(stopfile: str, tokens: Iterator[str])->Iterator[str]:
    with open(stopfile, 'r', encoding='utf-8') as f:
        stops = set([line.strip() for line in f])
    return (token for token in tokens if token not in stops)


def docs2xgram_docs(filename: str, xgram_model)->str:
    with open(filename, 'r', encoding='utf-8') as f:
        xgram_docs = ''
        for doc in nlp.pipe((line for line in f),
                            batch_size=2000, n_threads=3):
            doctokens = (token.lemma_ for token in doc)
            wostops = remove_stopwords('combined-stop-words.txt', doctokens)
            xgram_doc = ' '.join(xgram_model[list(wostops)])
            xgram_docs += xgram_doc
    return xgram_docs


if __name__ == '__main__':
    sentences = doc2sents(global_constants.SOURCE_FILE)
    save2file(global_constants.UNI_SENTS, sentences)
    bigram_model = xgram_model(global_constants.UNI_SENTS)
    save_model(global_constants.BI_MODEL, bigram_model)
    bigram_docs = docs2xgram_docs(global_constants.SOURCE_FILE, bigram_model)
    save2file(global_constants.BI_DOCS, bigram_docs)
    bigram_sentences = doc2sents(global_constants.BI_DOCS)
    save2file(global_constants.BI_SENTS, bigram_sentences)
