import os
import argparse
import fnmatch
from typing import Iterator, List
import gensim
from gensim import corpora
import time
import boto
import smart_open


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
    infiles = []
    for fname in fnmatch.filter(allfiles, matchfiles):
        if bucket:
            infile = '/'.join(['s3:/', bucket, fname])
            print(infile)
        else:
            infile = '/'.join(['s3:/', fname])
            print(infile)
            path, filename = os.path.split(infile)
        infiles.append(infile)
    return infiles


def make_dict(infiles: List[str]) -> gensim.corpora.Dictionary:
    start = time.time()
    term_dict = corpora.Dictionary([tokens for tokens in get_tokens(infiles)])
    if args.dictionary:
        with smart_open.smart_open(args.dictionary, 'wb') as fout:
            term_dict.save(fout)
    end = time.time()
    delta = end - start
    print('Time to Make Dictionary', delta, ' sec')
    return term_dict


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
    allfiles = list_files('pto-us-data', 'token-text')
    infiles = match_files(args.filename, allfiles, args.bucket)
    term_dict = make_dict(infiles)
    corpus = [term_dict.doc2bow(tokens)
              for tokens in get_tokens(infiles)]
    if args.vector:
        corpora.MmCorpus.serialize(args.vector, corpus, id2word=term_dict)
