import pickle
from datetime import datetime
from typing import List


def pickle_tokens(tokens: List[List[str]])->None:
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = ''.join(['pickle', timestamp, '.pkl'])
    with open(filename, 'wb') as f:
        pickle.dump(tokens, f)


def unpickle_tokens(filename: str)->List[List[str]]:
    with open(filename, 'rb') as f:
        tokens = pickle.load(f)
    return tokens
