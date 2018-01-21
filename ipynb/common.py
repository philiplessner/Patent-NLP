import os
import tarfile
import fnmatch
from typing import Iterator, Pattern
from cytoolz import curry


def file_find(filepat: str, top: str) -> Iterator[str]:
    ''' Find files with names that match filepat starting from directory top
        Parameters
            filepat: file pattern to match
            top: top level directory
        Returns
            String of full filepath to file
    '''
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            yield os.path.join(path, name)


def get_archivexml(filepath: str) -> Iterator[str]:
    '''Extract xml files from tar.gz archive one at a time
    Parameter
        filepath: full filepath to xml tar.gz archive
    Returns
        Iterator of utf-8 encoded xml string
    '''
    tar = tarfile.open(filepath)
    tarmems = tar.getmembers()
    for member in tarmems[1:]:
        f = tar.extractfile(member)
        yield f.read().decode(encoding='utf-8')
        f.close()
    tar.close()


@curry
def remove(pattern: Pattern[str], docstr: str) -> str:
    '''Replace a regular expression pattern with a space
    Parameters
        pattern: a compiled re pattern
        docstr: string to perform replacement on
    Returns
        string with pattern replaced by whitespace
    '''
    return pattern.sub(' ', docstr)
