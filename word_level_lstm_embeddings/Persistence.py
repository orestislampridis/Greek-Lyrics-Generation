import json


def save_vocabulary_info_json(vocabulary, path):
    with open(path, "w", encoding="UTF-8") as json_file:
        json.dump(vocabulary, json_file, ensure_ascii=False, indent=2)


def read_vocabulary_info_json(path):
    with open(path, "r", encoding="UTF-8") as json_file:
        vocabulary = json.load(json_file)
        return vocabulary
