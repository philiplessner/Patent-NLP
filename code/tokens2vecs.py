import sys
import os
import argparse
import logging
import fnmatch
from typing import Iterator, List, Tuple
import gensim
from gensim import corpora
import boto
import smart_open


Corpus = Iterator[Iterator[Tuple[int, int]]]


def get_tokens(infiles: List[str]) -> Iterator[str]:
    '''Yield list of strings from each line in each file
    Parameters
        infiles: list of full path strings to file containing strings
    Returns
        list of strings with newline stripped
    '''
    for infile in infiles:
        with smart_open.smart_open(infile, 'r', encoding='utf-8') as inf:
            for line in inf:
                yield line.strip().split()


def list_files(mybucket: str, prefix: str) -> List[str]:
    '''List files in a directory in an Amazon S3 Bucket
       Parameters
            mybucket: name of bucket
            prefix: directory in bucket
        Returns
            filepaths: list of files in bucket, prefix
    '''
    bucket = boto.connect_s3().get_bucket(mybucket)
    return [key.name for key in bucket.list(prefix=prefix)]


def match_files(matchfiles: str,
                allfiles: List[str], bucket: str) -> List[str]:
    '''Match a glob filename with a list of files
       Parameters
            matchfiles: glob filename
            allfiles: list of files to match
            bucket: Amazon S3 bucket name
        Returns
            infiles: list of files matched
    '''
    logger.info('%s', 'Input Files')
    infiles = []
    for fname in fnmatch.filter(allfiles, matchfiles):
        if bucket:
            infile = '/'.join(['s3:/', bucket, fname])
            logger.info('%s', infile)
        else:
            infile = '/'.join(['s3:/', fname])
            logger.info('%s', infile)
            path, filename = os.path.split(infile)
        infiles.append(infile)
    return infiles


def make_dict(infiles: List[str]) -> gensim.corpora.dictionary.Dictionary:
    '''Make a gensim corpora dictionary from files of newline terminated strings.
       Each newline terminated string is a document.
       Parameter
            infiles: list of filepaths
       Returns
            gensim corpora dictionary (mapping of integer id's to unique words)
    '''
    term_dict = corpora.Dictionary((tokens for tokens in get_tokens(infiles)))
    return term_dict


def make_corpus(infiles: List[str],
                term_dict: gensim.corpora.dictionary.Dictionary) -> Corpus:
    '''Make a gensim corpus from files of newline terminated strings and
       a gensim dictionary (made from the same files).
       The corpus is [(word id, word count), ...] for each document.
       Parameters
            infiles: list of filepaths
            term_dict: gensim corpora dictionary
        Returns
            Iterator used to construct corpus
            (corpus is Iterator[Iterator[Tuple[int, int]]])

    '''
    return (term_dict.doc2bow(tokens)
            for tokens in get_tokens(infiles))


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('../logs/nlp.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logstr = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(logstr)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    description = '''Vectorize tokens
                  '''
    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                     description=description)
    parser.add_argument('filename', help='glob of filenames')
    parser.add_argument('-b', '--bucket', help='S3 Bucket Name')
    parser.add_argument('-d', '--dictionary',
                        help='Full Path to Dictionary File')
    parser.add_argument('-v', '--vector',
                        help='Full Path to Vector File')
    args = parser.parse_args()
    logger = setup_logger()
    logger.info('%s%s', 'Running ', sys.argv[0])
    allfiles = list_files('pto-us-data', 'token-text')
    infiles = match_files(args.filename, allfiles, args.bucket)
    term_dict = make_dict(infiles)
    if args.dictionary:
        with smart_open.smart_open(args.dictionary, 'wb') as fout:
            term_dict.save(fout)
        logger.info('%s%s', 'Corpora Dictionary File ', args.dictionary)
    corpus = make_corpus(infiles, term_dict)
    if args.vector:
        corpora.MmCorpus.serialize(args.vector, corpus, id2word=term_dict)
        logger.info('%s%s', 'Corpus MM File ', args.vector)
