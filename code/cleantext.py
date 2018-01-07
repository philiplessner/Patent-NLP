import sys
import re
import os
from typing import Iterator, Pattern
from cytoolz import compose, curry


specialchar = re.compile(r'[#$%&()*+_/:<=>@^{}|\]\-\[]')
period = re.compile(r'\.(\w{2,})')
extra_punc = re.compile(r' [.,;] ')
special_word = re.compile(r'(su\w)\.')
numbers = re.compile(r'[0-9]+')
extrawhitespace = re.compile(r' {2,}')
singleletter = re.compile(r' [b-zB-Z] | [b-zB-Z][,;]')
nan = re.compile(r' nan$')


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


def tolowercase(docstr: str) -> str:
    return docstr.lower()


def remove_nonascii(s: str) -> str:
        return "".join(i for i in s if ord(i) < 128)


@curry
def remove_period(pattern: Pattern[str], docstr: str) -> str:
    '''Remove period from the end of some words leaving word
    Parameters
        pattern: a compilied re pattern with the word in group 1
                 and period outside of group
        docstr: string to perform replacement on
    Returns
        word without period at end
    '''
    return pattern.sub(r'\1', docstr)


def get_line(filename: str) -> Iterator[str]:
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()


def clean_text(filename: str) -> None:
    clean = compose(remove(extrawhitespace),
                    remove(extra_punc),
                    remove_period(special_word),
                    remove_period(period),
                    remove(singleletter),
                    remove(nan),
                    remove(numbers),
                    remove(specialchar),
                    remove_nonascii)
    fileroot = os.path.splitext(os.path.split(filename)[1])[0]
    with open(fileroot + '_clean' + '.txt', 'w', encoding='utf-8') as f:
        for doc in get_line(filename):
            f.write(clean(doc) + '\n')


if __name__ == '__main__':
    filename = sys.argv[1]
    clean_text(filename)
