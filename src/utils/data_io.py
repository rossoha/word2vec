from os import listdir
from os.path import isfile, join


def load_data(data_path):
    with open(data_path, 'r') as file:
        data = file.readlines()
    return data


def list_files(base_path):
    return [f for f in listdir(base_path) if isfile(join(base_path, f))]


def save_data(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(data)
