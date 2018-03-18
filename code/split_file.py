import argparse
import os
from typing import Iterator, Optional
import smart_open


def get_doc(infile: str) -> Iterator[str]:
    '''Yield new line terminated strings
    Parameters
        infile: full path string to file containing strings
    Returns
        each string
    '''
    with smart_open.smart_open(infile, 'r', encoding='utf-8') as inf:
        for line in inf:
            yield line


def file2multi(infile: str, bucket: Optional[str],
               outfile_path: Optional[str]) -> None:
    count = 0
    if bucket:
        infile = '/'.join(['s3:/', bucket, infile])
    if outfile_path:
        path = '/'.join(['s3:/', bucket, outfile_path])
        _, filename = os.path.split(infile)
    else:
        path, filename = os.path.split(infile)
    print('Reading From:', infile)
    root = os.path.splitext(filename)[0]
    outfile = os.path.join(path, '{:02d}'.format(count) + root + '.txt')
    f = smart_open.smart_open(outfile, 'w', encoding='utf-8')
    for index, doc in enumerate(get_doc(infile)):
        if (index % 10000):
            f.write(doc)
        else:
            f.close()
            count += 1
            outfile = os.path.join(path, '{:02d}'.format(count) + root + '.txt')
            print('Writing to:', outfile, '\n')
            f = smart_open.smart_open(outfile, 'w', encoding='utf-8')
            f.write(doc)
    f.close()


if __name__ == '__main__':
    description = '''Split a raw text file into
                     sequentially numbered separate files.
                  '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description)
    parser.add_argument('filename')
    parser.add_argument('-b', '--bucket', help='S3 Bucket Name')
    parser.add_argument('-o', '--output', help='Output Path')
    args = parser.parse_args()
    file2multi(args.filename, args.bucket, args.output)
