import os
import argparse
import fnmatch
from typing import Iterator, List
import gensim
from gensim import corpora, models
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
    args = parser.parse_args()
    filepaths = list_file('pto-us-data', 'token-text')
    start = time.time()
    term_dict = corpora.Dictionary(documents=None)
    for fname in fnmatch.filter(filepaths, args.filename):
        if args.bucket:
            infile = '/'.join(['s3:/', args.bucket, fname])
            print(infile)
        else:
            infile = '/'.join(['s3:/', fname])
            print(infile)
            path, filename = os.path.split(infile)
        term_dict.add_documents([tokens for tokens in get_tokens(infile)])
    term_dict.compactify()
    end = time.time()
    delta = end - start
    print('Time to Make Dictionary', delta, ' sec')
