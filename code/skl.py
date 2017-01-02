from typing import List
from toolz import compose
from toolz.curried import get
from todisk import unpickle_tokens
from get_tokens import skl_analyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def tokens2strings(filename):
    all_tokens = unpickle_tokens(filename)
    tokensasstrings = [' '.join(tokens) for tokens in all_tokens]
    return tokensasstrings


def docs2strs(filename: str)->List[str]:
    '''Return a list of strings--one doc per string,  from a
       file where each document is a newline
       terminated string.
    '''
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
              for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


if __name__ == '__main__':
    pat = CountVectorizer(analyzer=skl_analyzer)
    lda = LatentDirichletAllocation(n_topics=50)
    compose(lda.fit,
            pat.fit_transform,
            get([0, 1000]),
            docs2strs)('plaintext.txt')
    print("\nTopics in LDA model:")
    feature_names = pat.get_feature_names()
    print_top_words(lda, feature_names, 20)
