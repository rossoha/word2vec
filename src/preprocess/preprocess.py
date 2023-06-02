from src.preprocess.text.cleaning import clean, remove_stop_words
from src.preprocess.text.staming import Steaming


def preprocess_text(text, rm_stop_words: bool = False):
    # Convert to lowercase
    # Remove stop words and  text to lowercase
    # Remove punctuation and digits
    text = clean(text)
    # Remove stopwords
    if rm_stop_words:
        text = remove_stop_words(text)
    # Steam text
    steamed = Steaming.stem_text(text)
    return steamed


def preprocess_dataset(dataset, rm_stop_words: bool = False):
    # Preprocess each text in the dataset
    preprocessed_dataset = []
    for text in dataset:
        preprocessed_text = preprocess_text(text=text, rm_stop_words=rm_stop_words)
        preprocessed_dataset.append(preprocessed_text)

    return preprocessed_dataset
