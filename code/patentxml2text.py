import re
import string
import xml.etree.ElementTree as ET
from itertools import chain
from typing import List, Iterator
# import json
from cytoolz import compose
import global_constants


def get_indivdocs(filepath: str) -> Iterator[str]:
    '''Split a file contanining multiple xml docs into a list that contains
       each xml doc as a string.
       Parameter
            filepath: full path string to file containing multiple xml docs
       Returns
            iterator of strs with each string being an individual xml document
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


def patent_type(doc: str, patenttype_tocheck: str) -> bool:
    '''Check whether the patent xml document is the type we want to collect
    Parameters
        doc: str containing xml document
        patenttype_tocheck: utility, design, plant
    Returns
        True if patent type is the type we want
        False if the type filed does not exist or if it is a different type
    '''
    root = ET.fromstring(doc)
    bib = root.findall('us-bibliographic-data-grant')
    if not bib:
        return False
    return (True if bib[0][1].attrib['appl-type'] == patenttype_tocheck
            else False)


def filter_patents(filepath: str)->Iterator[str]:
    '''Filter the xml documents so only the docs with the
    type of patent we want is returned.
    Parameters
        filepath: full filepath string to xml documents
    Returns
    an iterator of the document string
    '''
    for doc in get_indivdocs(filepath):
        if patent_type(doc, 'utility'):
            yield doc


def xml2plaintext(doc: str) -> Iterator[str]:
    '''Convert the xml documents into plain text from selected tags
    Parameters
        filepath: full filepath string to xml documents
    Returns
        an iterator of plaintext document strings
    '''
    tags_toget = ['abstract', 'claims']
    return chain.from_iterable((''.join(child.itertext()).splitlines()
                                for child in ET.fromstring(doc)
                                if child.tag in tags_toget))


def get_plaintext(filepath: str) -> List[List[str]]:
    '''Get plaintext documents from XML for type of document and fields wanted.
    Parameters
        filepath: full filepath string to XML file
    Returns
        A list containing lists of plaintext strings (one for each document)
    '''
    filtered_docs = (doc for doc in get_indivdocs(filepath)
                     if patent_type(doc, 'utility'))
    return [list(xml2plaintext(doc)) for doc in filtered_docs]


def filter_unneededstr(docs: List[List[str]]) -> List[List[str]]:
    '''Filter out blank strings ("").
    Parameters
        docs: iterator of iterator of strings
    Returns
        list of list of strings with "" removed
        each list is a document
    '''
    return [[s for s in doc if s != ''] for doc in docs]


def tosinglestr(docs: List[List[str]]) -> str:
    for doc in docs:
        doc.append('\n')
    return ''.join([' '.join(doc) for doc in docs])


def remove_allcaps(docstr: str) -> str:
    return re.sub('[A-Z]{2,}', ' ', docstr)


def remove_numbers(docstr: str) -> str:
    return re.sub('[0-9]+', ' ', docstr)


def remove_punctuation(docstr: str) -> str:
    table = str.maketrans({key: None for key in string.punctuation})
    return docstr.translate(table)


def remove_specialchar(docstr: str) -> str:
    pattern = re.compile(r'[#$%&()*+_/:<=>@^{}|] | \[ | ] | -')
    return pattern.sub(" ", docstr)


def tolowercase(docstr: str) -> str:
    return docstr.lower()


def remove_extrawhitespace(docstr: str) -> str:
    return re.sub(' {2,}', ' ', docstr)


def remove_nonascii(s: str) -> str:
        return "".join(i for i in s if ord(i) < 128)


def str2file(filepath: str, docstr: str) -> None:
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


def get_patentnumbers(filepath: str) -> Iterator[str]:
    for doc in filter_patents(filepath):
        yield str(int(ET.fromstring(doc).findall('.//doc-number')[0].text))


def patentnumber2file(infile: str, outfile: str) -> None:
    with open(outfile, 'w', encoding='utf-8') as of:
        for doc in get_patentnumbers(infile):
            of.write(doc + '\n')


def get_classifications(filepath: str) -> Iterator[str]:
    '''Get the CPC classfication
    Parameter
        filepath: full path to file containing xml
    Returns
        iterator of patent classfication strings, one per patent
    '''
    for doc in filter_patents(filepath):
        root = ET.fromstring(doc)
        yield ''.join([root.findall('.//section')[0].text,
                       root.findall('.//class')[0].text,
                       root.findall('.//subclass')[0].text])


def classification2file(infile: str, outfile: str) -> None:
    with open(outfile, 'w', encoding='utf-8') as of:
        for doc in get_classifications(infile):
            of.write(doc + '\n')


if __name__ == '__main__':
    filepath = global_constants.XML_FILE
    # with open('replacements.json', 'r', encoding='utf-8') as f:
    # replace_dictionary = json.load(f)
    # replace_patterns = [(k, v) for k, v in replace_dictionary[0].items()]
    # elements_replacer = RegexpReplacer(patterns=replace_patterns)
    docs_woxml = compose(remove_extrawhitespace,
                         # elements_replacer.replace,
                         remove_specialchar,
                         remove_nonascii,
                         remove_numbers,
                         remove_allcaps,
                         tosinglestr,
                         filter_unneededstr,
                         get_plaintext)(filepath)
    str2file(global_constants.SOURCE_FILE, docs_woxml)
    patentnumber2file(filepath, global_constants.PATNUM_FILE)
    classification2file(filepath, global_constants.CLASS_FILE)
