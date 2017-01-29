import re
from cytoolz import compose
import global_constants
from utilities import read_file, save2file


def remove_dualgrams(tobe_cleaned: str)->str:
    return re.sub(r'(?P<first>[a-z]+)_(?P=first)', r'\1 \1', tobe_cleaned)


def remove_singlechar(tobe_cleaned: str)->str:
    return re.sub(r' [a-z] ', ' ', tobe_cleaned)


if __name__ == '__main__':
    cleaned = compose(remove_singlechar,
                      remove_dualgrams,
                      remove_dualgrams,
                      read_file)(global_constants.TRI_SENTS)
    save2file('../intermediate/trigram_sentences_cleaned.txt', cleaned)
