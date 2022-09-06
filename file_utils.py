import yaml


def read_yaml(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


def read_texts(path):
    texts = list()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def split_text_word(texts):
    texts = [line.split() for line in texts]
    return texts
