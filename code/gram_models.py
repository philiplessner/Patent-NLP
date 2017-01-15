from typing import Iterator, List
from toolz import compose, curry
from toolz.curried import do
import spacy
import gensim
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
import global_constants
from utilities import save2file, save_model


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


def get_doc(infile: str) -> Iterator[str]:
    with open(infile, 'r', encoding='utf-8') as inf:
        for line in inf:
            yield line.strip()


def get_sentence(infile: str) -> Iterator[str]:
    for doc in nlp.pipe(get_doc(infile),
                        batch_size=5000, n_threads=-1):
        for sent in doc.sents:
            yield ' '.join((token.lemma_
                            for token in sent
                            if not token.is_punct)) + '\n'


@curry
def doc2sents2(outfile: str, infile: str) -> str:
        with open(outfile, 'w', encoding='utf-8') as outf:
            for newsent in get_sentence(infile):
                outf.write(newsent)
        return outfile


def doc2processed_doc(filename: str)->str:
    with open(filename, 'r', encoding='utf-8') as f:
        documents = []  # type: List[str]
        for doc in nlp.pipe((line for line in f),
                            batch_size=5000, n_threads=-1):
            newdoc = ' '.join((token.lemma_
                               for token in doc
                               if not token.is_punct))
            documents.append(newdoc)
    return ''.join(documents)


def xgram_model(filename: str)->gensim.models.phrases.Phrases:
    return Phraser(Phrases(LineSentence(filename)))


def remove_stopwords(stopfile: str, tokens: Iterator[str])->Iterator[str]:
    with open(stopfile, 'r', encoding='utf-8') as f:
        stops = set([line.strip() for line in f])
    return (token for token in tokens if token not in stops)


@curry
def xgram_strings(filename: str,
                  xgram_model: gensim.models.phrases.Phraser)->str:
    return '\n'.join(' '.join(xgram_model[sentences])
                     for sentences in LineSentence(filename))


if __name__ == '__main__':
    nlp = spacy.load('en')
    xgram_pipe = compose(save2file(global_constants.TRI_SENTS),
                         xgram_strings(global_constants.BI_SENTS),
                         do(save_model(global_constants.TRI_MODEL)),
                         xgram_model,
                         save2file(global_constants.BI_SENTS),
                         xgram_strings(global_constants.UNI_SENTS),
                         do(save_model(global_constants.BI_MODEL)),
                         xgram_model,
                         # save2file(global_constants.UNI_SENTS),
                         doc2sents2(global_constants.UNI_SENTS))
    xgram_pipe(global_constants.SOURCE_FILE)
    # lemmatized = doc2processed_doc(global_constants.SOURCE_FILE)
    # doc2sents2(global_constants.SOURCE_FILE, '../intermediate/test.txt')
