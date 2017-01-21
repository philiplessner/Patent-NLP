from typing import Iterator
import multiprocessing
from toolz import compose, curry
from toolz.curried import do
import spacy
import gensim
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
import global_constants
from utilities import save_model


def get_doc(infile: str) -> Iterator[str]:
    with open(infile, 'r', encoding='utf-8') as inf:
        for line in inf:
            yield line.strip()


def get_doc2(infile: str) -> Iterator[str]:
    with open(infile, 'r', encoding='utf-8') as inf:
        for line in inf:
            yield line


def process_sentence(infile: str) -> Iterator[str]:
    for doc in nlp.pipe(get_doc(infile),
                        batch_size=5000,
                        n_threads=multiprocessing.cpu_count()):
        for sent in doc.sents:
            yield ' '.join((token.lemma_
                            for token in sent
                            if not token.is_punct)) + '\n'


def process_document(infile: str) -> Iterator[str]:
    for doc in nlp.pipe(get_doc(infile),
                        batch_size=5000,
                        n_threads=multiprocessing.cpu_count()):
        yield ' '.join((token.lemma_
                        for token in doc
                        if not token.is_punct))


@curry
def doc2processed_doc(outfile: str, infile: str) -> str:
    with open(outfile, 'w', encoding='utf-8') as outf:
        for processed_doc in process_document(infile):
            outf.write(processed_doc)
    return outfile


@curry
def doc2sents(outfile: str, infile: str) -> str:
    with open(outfile, 'w', encoding='utf-8') as outf:
        for sentence in process_sentence(infile):
            outf.write(sentence)
    return outfile


def xgram_model(filename: str)->gensim.models.phrases.Phrases:
    return Phraser(Phrases(LineSentence(filename)))


def remove_stopwords(stopfile: str, tokens: Iterator[str])->Iterator[str]:
    with open(stopfile, 'r', encoding='utf-8') as f:
        stops = set([line.strip() for line in f])
    return (token for token in tokens if token not in stops)


def xgram_strings(filename: str,
                  xgram_model: gensim.models.phrases.Phraser) -> Iterator[str]:
    for sentences in LineSentence(filename):
        yield ' '.join(xgram_model[sentences])


@curry
def xgrams2file(outfile: str,
                infile: str,
                xgram_model: gensim.models.phrases.Phraser) -> str:
    with open(outfile, 'w', encoding='utf-8') as outf:
        for gramsent in xgram_strings(infile, xgram_model):
            outf.write(gramsent)
    return outfile


if __name__ == '__main__':
    nlp = spacy.load('en')
    xgram_pipe = compose(xgrams2file(global_constants.TRI_SENTS,
                                     global_constants.BI_SENTS),
                         do(save_model(global_constants.TRI_MODEL)),
                         xgram_model,
                         xgrams2file(global_constants.BI_SENTS,
                                     global_constants.UNI_SENTS),
                         do(save_model(global_constants.BI_MODEL)),
                         xgram_model,
                         doc2sents(global_constants.UNI_SENTS))
    # xgram_pipe(global_constants.SOURCE_FILE)
    doc2processed_doc('../intermediate/abstractclaims_lem.txt',
                      global_constants.SOURCE_FILE)
