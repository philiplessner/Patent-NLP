import sys
import os
from typing import Iterator
import multiprocessing
import spacy


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


def process_sentence(infile: str) -> Iterator[str]:
    ''' Split a newline terminated string(document) into sentences using spacy
    Parameters
        infile: text file containing newline terminated strings
    Returns
        Iterator of lower cased newline terminated sentences

    '''
    for doc in nlp.pipe(get_doc(infile),
                        batch_size=10000,
                        n_threads=multiprocessing.cpu_count()):
        for sent in doc.sents:
            yield ' '.join((token.lower_
                            for token in sent
                            if not token.is_punct)) + '\n'


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


if __name__ == '__main__':
    nlp = spacy.load('en')
    infile = sys.argv[1]
    path, filename = os.path.split(infile)
    root = os.path.splitext(filename)[0]
    outfile = os.path.join(path, root + '_uni_sents.txt')
    doc2sents(outfile, infile)
