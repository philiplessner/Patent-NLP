import sys
import os
from typing import Iterator
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


def file2multi(infile: str) -> None:
    count = 0
    path, filename = os.path.split(infile)
    root = os.path.splitext(filename)[0]
    outfile = os.path.join(path, root + str(count) + '.txt')
    f = smart_open.smart_open(outfile, 'w', encoding='utf-8')
    for index, doc in enumerate(get_doc(infile)):
        if (index % 10000):
            f.write(doc)
        else:
            f.close()
            count += 1
            outfile = os.path.join(path, root + str(count) + '.txt')
            print('Writing to:', outfile, '\n')
            f = smart_open.smart_open(outfile, 'w', encoding='utf-8')
            f.write(doc)
    f.close()


if __name__ == '__main__':
    infile = sys.argv[1]
    file2multi(infile)
