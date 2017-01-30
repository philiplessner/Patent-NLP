import re
import string
import xml.etree.ElementTree as ET
from itertools import chain
from typing import List, Iterator
import json
from cytoolz import compose
import global_constants


def get_indivdocs(filepath: str)->List[str]:
    '''Split a file contanining multiple xml docs into a list that contains
       each xml doc as a string.
       Parameter
            filepath: path to file containing multiple xml docs
       Returns
            list of strs with each string being an individual xml document
    '''
    d = []
    s = ""
    with open(filepath, 'r') as f:
        for l in f:
            if l == '<?xml version="1.0" encoding="UTF-8"?>\n':
                if len(s) > 0:
                    d.append(s)
                s = ""
            s += l
        d.append(s)
    return d


def get_indivdocs2(filepath: str) -> Iterator[str]:
    '''Split a file contanining multiple xml docs into a list that contains
       each xml doc as a string.
       Parameter
            filepath: path to file containing multiple xml docs
       Returns
            list of strs with each string being an individual xml document
    '''
    s = ""
    with open(filepath, 'r') as f:
        for l in f:
            if l == '<?xml version="1.0" encoding="UTF-8"?>\n':
                if len(s) > 0:
                    yield s
                s = ""
            s += l
        yield s


def patent_type(doc: str, patenttype_tocheck: str)->bool:
    root = ET.fromstring(doc)
    bib = root.findall('us-bibliographic-data-grant')
    if not bib:
        return False
    return (True if bib[0][1].attrib['appl-type'] == patenttype_tocheck
            else False)


def filter_patents(docs: List[str])->Iterator[str]:
    return (doc for doc in docs if patent_type(doc, 'utility'))


def filter_patents2(filepath: str)->Iterator[str]:
    for doc in get_indivdocs2(filepath):
        if patent_type(doc, 'utility'):
            yield doc


def xml2plaintext(raw_xml: Iterator[str])->Iterator[Iterator[str]]:
    tags_toget = ['abstract', 'claims']
    return (chain.from_iterable((''.join(child.itertext()).splitlines()
                                 for child in ET.fromstring(doc)
                                 if child.tag in tags_toget))
            for doc in raw_xml)


def get_patentnumbers(filepath: str) -> Iterator[str]:
    for doc in filter_patents2(filepath):
        yield str(int(ET.fromstring(doc).findall('.//doc-number')[0].text))


def patentnumber2file(infile: str, outfile: str) -> None:
    with open(outfile, 'w', encoding='utf-8') as of:
        for doc in get_patentnumbers(infile):
            of.write(doc + '\n')


def filter_unneededstr(docs: Iterator[Iterator[str]])->List[List[str]]:
    return [[s for s in doc if s != ''] for doc in docs]


def tosinglestr(docs: List[List[str]])->str:
    for doc in docs:
        doc.append('\n')
    return ''.join([' '.join(doc) for doc in docs])


def remove_allcaps(docstr: str)->str:
    return re.sub('[A-Z]{2,}', ' ', docstr)


def remove_numbers(docstr: str)->str:
    return re.sub('[0-9]+', ' ', docstr)


def remove_punctuation(docstr: str)->str:
    table = str.maketrans({key: None for key in string.punctuation})
    return docstr.translate(table)


def tolowercase(docstr: str)->str:
    return docstr.lower()


def remove_extrawhitespace(docstr: str)->str:
    return re.sub(' {2,}', ' ', docstr)


def remove_nonascii(s: str)->str:
        return "".join(i for i in s if ord(i) < 128)


def str2file(filepath: str, docstr: str)->None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(docstr)


class RegexpReplacer():
    '''
    Replaces regular expression in a text.
    '''
    def __init__(self, patterns=None):
        self.__patterns = [(re.compile(regex), repl)
                           for (regex, repl) in patterns]

    def replace(self, text):
        s = text

        for (pattern, repl) in self.__patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s


if __name__ == '__main__':
    filepath = global_constants.XML_FILE
    with open('replacements.json', 'r', encoding='utf-8') as f:
        replace_dictionary = json.load(f)
    replace_patterns = [(k, v) for k, v in replace_dictionary[0].items()]
    elements_replacer = RegexpReplacer(patterns=replace_patterns)
    docs_woxml = compose(remove_extrawhitespace,
                         elements_replacer.replace,
                         remove_nonascii,
                         remove_numbers,
                         remove_allcaps,
                         tosinglestr,
                         filter_unneededstr,
                         xml2plaintext,
                         filter_patents,
                         get_indivdocs)(filepath)
    str2file(global_constants.SOURCE_FILE, docs_woxml)
