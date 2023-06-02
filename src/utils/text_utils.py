import re
import string


def build_vocabulary(data):
    vocabulary = set()
    for sentence in data:
        vocabulary.update(sentence)
    return list(vocabulary)


def build_corpus(data):
    corpus = []
    for sentence in data:
        sentence_words = [remove_extra_whitespaces(word) for word in sentence.split(" ")]
        corpus.append(sentence_words)
    return corpus


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = translate(text)

    # Remove numbers
    text = remove_numbers(text)

    # Remove extra whitespaces
    text = remove_extra_whitespaces(text)

    # Remove leading/trailing whitespaces
    text = remove_leading_or_trailing_whitespaces(text)

    return text

    # Removes punctuation with RegExp """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""


def translate(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def remove_extra_whitespaces(text):
    return re.sub(r'\s+', ' ', text)


def remove_leading_or_trailing_whitespaces(text):
    return text.strip()
