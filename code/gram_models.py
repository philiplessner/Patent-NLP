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
    '''Yield new line terminated strings with newlines stripped
    Parameters
        infile: full path string to file containing strings
    Returns
        each string with newline stripped
    '''
    with open(infile, 'r', encoding='utf-8') as inf:
        for line in inf:
            yield line.strip()


def get_doc2(infile: str) -> Iterator[str]:
    with open(infile, 'r', encoding='utf-8') as inf:
        for line in inf:
            yield line


def process_sentence(infile: str) -> Iterator[str]:
    ''' Split a newline terminated string(document) into sentences using spacy
    Parameters
        infile: text file containing newline terminated strings
    Returns
        Iterator of lematized newline terminated sentences

    '''
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
    '''Write sentences to a file. These are unigram sentences
       used for further processing.
    Parameters
        infile: text file containing newline terminated document strings
        outfile: text file containing newline terminated sentence strings
    Returns
        outfile: string full path of outfile to pass through
    '''
    with open(outfile, 'w', encoding='utf-8') as outf:
        for sentence in process_sentence(infile):
            outf.write(sentence)
    return outfile


def xgram_model(filename: str)->gensim.models.phrases.Phrases:
    '''Make a n+1-gram  model from a file containing n-gram sentences.
    Parameters
        filename: full path string of filename containing n-gram sentences
    Returns
        n+1-gram model
    '''
    return Phraser(Phrases(LineSentence(filename)))


def xgram_strings(filename: str,
                  xgram_model: gensim.models.phrases.Phraser) -> Iterator[str]:
    '''From a file containing n-gram sentences and a n+1-gram model
       generate n+1-gram sentences.
    Parameters
        filename: full path string to file containing n-gram sentences
        xgram_model: model for n+1-grams
    Returns
        Iterator for n+1-gram sentences
    '''
    for sentences in LineSentence(filename):
        yield ' '.join(xgram_model[sentences]) + '\n'


@curry
def xgrams2file(outfile: str,
                infile: str,
                xgram_model: gensim.models.phrases.Phraser) -> str:
    '''Write xgram sentences to file.
    Parameters
        outfile: full path string to file to write n+1-gram sentences to
        infile: full path string to file containing n-gram sentences
        xgram_model: model for n+1-grams
    Returns
         outfile: full path string to n+1-gram file
    '''
    with open(outfile, 'w', encoding='utf-8') as outf:
        for gramsent in xgram_strings(infile, xgram_model):
            outf.write(gramsent)
    return outfile


def remove_stopwords(stopfile: str, tokens: Iterator[str])->Iterator[str]:
    with open(stopfile, 'r', encoding='utf-8') as f:
        stops = set([line.strip() for line in f])
    return (token for token in tokens if token not in stops)


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
    xgram_pipe(global_constants.SOURCE_FILE)
    # doc2processed_doc('../intermediate/abstractclaims_lem.txt',
    # global_constants.SOURCE_FILE)
