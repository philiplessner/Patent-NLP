import argparse
from gensim import corpora, models
from gensim.similarities.docsim import SparseMatrixSimilarity


if __name__ == '__main__':
    description = '''Convert TfIdf Vectors
                     to Similarity Matrix
                  '''
    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class,
                                     description=description)
    parser.add_argument('filename', help='path to TfIdf Model File')
    parser.add_argument('-i', '--index',
                        help='Full Path to Index File')
    parser.add_argument('-d', '--dictionary',
                        help='Full Path to Dictionary File')
    parser.add_argument('-v', '--vector',
                        help='Full Path to Corpus Vector File')
    args = parser.parse_args()
    tfidf_model = models.TfidfModel.load(args.filename)
    corpus = corpora.MmCorpus(args.vector)
    corpus_tfidf = tfidf_model[corpus]
    corpora_dict = corpora.Dictionary.load(args.dictionary)
    tfidf_index = SparseMatrixSimilarity(corpus_tfidf,
                                         num_features=len(corpora_dict))
    tfidf_index.save(args.index)
