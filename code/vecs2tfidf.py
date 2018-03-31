import sys
import logging
import argparse
from gensim import corpora, models


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('../logs/nlp.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logstr = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(logstr)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    description = '''Convert Corpus Count Vectors
                     to TfIdf Vectors
                  '''
    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                     description=description)
    parser.add_argument('filename', help='path to corpus MM format file')
    parser.add_argument('-d', '--dictionary',
                        help='Full Path to Dictionary File')
    parser.add_argument('-t', '--tfidf',
                        help='Full Path to TfIdf Model File')
    args = parser.parse_args()
    logger = setup_logger()
    logger.info('%s%s', 'Running ', sys.argv[0])
    corpus = corpora.MmCorpus(args.filename)
    if args.dictionary:
        corpora_dict = corpora.Dictionary.load(args.dictionary)
    tfidf_model = models.TfidfModel(corpus,
                                    id2word=corpora_dict,
                                    dictionary=corpora_dict,
                                    normalize=True)
    tfidf_model.save(args.tfidf)
    corpus_tfidf = tfidf_model[corpus]
