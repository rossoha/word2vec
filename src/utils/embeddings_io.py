import pickle

def save_embedding(embedding_matrix, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(embedding_matrix, f)

def load_embedding(filepath):
    with open(filepath, 'rb') as f:
        embedding_matrix = pickle.load(f)
    return embedding_matrix

