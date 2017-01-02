from itertools import islice


def save2file(filepath: str, tosave: str)->None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(tosave)


def read_file(filepath: str)->str:
    with open(filepath, 'r', encoding='utf-8') as f:
        contents = f.read()
    return contents


def lines_from_file(filepath: str, line: int)->str:
    with open(filepath, 'r', encoding='utf-8') as f:
        contents = list(islice(f, line, line + 1))
    return contents[0]


def save_model(filepath: str, model)->None:
    model.save(filepath)
