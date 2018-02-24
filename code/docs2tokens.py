#!/usr/bin/env python
import os
import multiprocessing
import argparse
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
    with smart_open.smart_open(infile, 'r', encoding='utf-8') as inf:
        for line in inf:
            yield line.strip()


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
    with smart_open.smart_open(outfile, 'w', encoding='utf-8') as outf:
        for tokens in doc2tokens(infile):
            outf.write(tokens + '\n')
    return outfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize raw text lemmatize and strip punctuation')
    parser.add_argument('filename', nargs='+')
    parser.add_argument('-b', '--bucket', help='S3 Bucket Name')
    parser.add_argument('-o', '--output', help='Output Path')
    args = parser.parse_args()
    nlp = spacy.load('en_core_web_sm')
    for fname in args.filename:
        if args.bucket:
            infile = '/'.join(['s3:/', args.bucket, fname])
            print(infile)
        else:
            infile = '/'.join(['s3:/', fname])
            print(infile)
        if args.output:
            _, filename = os.path.split(infile)
            root = os.path.splitext(filename)[0]
            outfile = os.path.join('s3://',
                                   args.bucket,
                                   args.output,
                                   root + '_tokens.txt')
            print(outfile)
        else:
            path, filename = os.path.split(infile)
            root = os.path.splitext(filename)[0]
            outfile = os.path.join(path, root + '_tokens.txt')
            print(outfile)
        doctokens2file(infile, outfile)
