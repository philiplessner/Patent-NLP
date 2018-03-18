import os
import argparse
import fnmatch
from typing import Iterator, List
from gensim import corpora
import time
import boto
import smart_open


def get_tokens(infile: str) -> Iterator[str]:
    '''Yield new line terminated strings with newlines stripped
    Parameters
        infile: full path string to file containing strings
    Returns
        each string with newline stripped
    '''
    with smart_open.smart_open(infile, 'r', encoding='utf-8') as inf:
        for line in inf:
            yield line.strip().split()


def list_file(mybucket: str, prefix: str) -> List[str]:
    '''List files in an Amazon S3 Bucket
       Parameters
            mybucket: name of bucket
            prefix: directory in bucket
        Returns
            filepaths: list of files in bucket, prefix
    '''
    bucket = boto.connect_s3().get_bucket(mybucket)
    return [key.name for key in bucket.list(prefix=prefix)]


if __name__ == '__main__':
    description = '''Vectorize tokens
                  '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('filename')
    parser.add_argument('-b', '--bucket', help='S3 Bucket Name')
    parser.add_argument('-d', '--dictionary',
                        help='Full Path to Dictionary File')
    parser.add_argument('-v', '--vector',
                        help='Full Path to Vector File')
    args = parser.parse_args()
    filepaths = list_file('pto-us-data', 'token-text')
    infiles = []
    for fname in fnmatch.filter(filepaths, args.filename):
        if args.bucket:
            infile = '/'.join(['s3:/', args.bucket, fname])
            print(infile)
        else:
            infile = '/'.join(['s3:/', fname])
            print(infile)
            path, filename = os.path.split(infile)
        infiles.append(infile)
    start = time.time()
    term_dict = corpora.Dictionary([tokens for tokens in get_tokens(infile)
                                    for infile in infiles])
    if args.dictionary:
        with smart_open.smart_open(args.dictionary, 'wb') as fout:
            term_dict.save(fout)
    end = time.time()
    delta = end - start
    print('Time to Make Dictionary', delta, ' sec')
    corpus = [term_dict.doc2bow(tokens)
              for tokens in get_tokens(infile) for infile in infiles]
    if args.vector:
        corpora.MmCorpus.serialize(args.vector, corpus, id2word=term_dict)
