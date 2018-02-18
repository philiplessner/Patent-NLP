import multiprocessing
from typing import Iterator
import spacy
import smart_open


def get_doc(infile: str) -> Iterator[str]:
    '''Yield new line terminated strings with newlines stripped
    Parameters
        infile: full path string to file containing strings
    Returns
        each string with newline stripped
    '''
    with smart_open.smart_open(infile) as inf:
        for line in inf:
            yield line.decode('utf-8').strip()


def doc2tokens(infile: str) -> Iterator[str]:
    '''Lemantize and remove punctuation from strings (documents)
    Parameters
        infile: full string path to file with newline terminated strings
    Returns
        an iterator of lematized strings with punctuation removed

    '''
    for doc in nlp.pipe(get_doc(infile),
                        batch_size=1000,
                        n_threads=multiprocessing.cpu_count()):
        yield ' '.join((token.lemma_
                        for token in doc
                        if not token.is_punct))


def doctokens2file(infile: str, outfile: str) -> str:
    '''Write lematized strings (documents) with punctuation removed
    to file. Each document is a newline separated string.
    Parameters
        infile: full string path to original strings
        outfile: full string path to file to write processed strings
    Returns
        outfile: full string path to file with processed strings
    '''
    with smart_open.smart_open(outfile, 'wb') as outf:
        for tokens in doc2tokens(infile):
            outf.write(tokens.encode('utf-8') + '\n'.encode('utf-8'))
    return outfile


if __name__ == '__main__':
    # File String Constants
    INPUTFILE = 's3://pto-us-data/text-data/titleabstract03.txt'
    OUTPUTFILE = 's3://pto-us-data/text-data/titleabstract_tokens03.txt'
    nlp = spacy.load('en_core_web_sm')
    doctokens2file(INPUTFILE, OUTPUTFILE)
