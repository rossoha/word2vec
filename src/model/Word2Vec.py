import datetime as dt
import math
import os
import pickle
import sys
import time
from typing import List

import numpy as np
import pandas as pd
from boltons.setutils import IndexedSet
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class Word2Vec:
    def __init__(self, window_size, embedding_size, learning_rate):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.word2index = {}
        self.index2word = {}
        self.vocab_size = 0
        self.W_in = None
        self.W_out = None
        self.losses = []
        self.vocab: IndexedSet[str] = IndexedSet()

    def build_vocab(self, corpus):
        vocab = set()
        for sentence in corpus:
            for word in sentence:
                if word not in vocab:
                    vocab.add(word)
        indexed_set = IndexedSet(vocab)
        self.vocab = indexed_set
        self.vocab_size = len(vocab)
        print(f"Vocabulary size is: {self.vocab_size}")

    def initialize_weights(self):
        self.W_in = np.random.uniform(-0.8, 0.8, (self.vocab_size, self.embedding_size))
        self.W_out = np.random.uniform(-0.8, 0.8, (self.embedding_size, self.vocab_size))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def forward_propagation(self, center_words, context_words):
        raise ValueError("Not implemented.")

    def backward_propagation(self, center_word, context_words, output_probs):
        raise ValueError("Not implemented.")

    def cosine_similarity(self, word1, word2):
        word_index1 = self.word2index[word1]
        word_index2 = self.word2index[word2]
        embedding1 = self.W_in[word_index1]
        embedding2 = self.W_in[word_index2]
        dot_product = np.dot(embedding1, embedding2)
        norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        similarity_score = dot_product / norm_product
        return similarity_score

    def cluster_words(self, num_clusters, path=None, show=False):
        raise ValueError("Not implemented.")

    def calculate_loss(self, output_probs, center_words, context_words):
        raise ValueError("Not implemented.")

    def add_loss(self, loss):
        self.losses.append(loss)

    def avg_loss(self):
        return round(math.fsum(self.losses) / len(self.losses), 3)

    def get_context_window(self, sentence: List[str], index: int) -> List[str]:
        context_window = []
        start = max(0, index - self.window_size)
        end = min(len(sentence), index + self.window_size + 1)
        for i in range(start, end):
            if i != index:
                context_window.append(sentence[i])
        return context_window

    def save_model(self, file_path):
        print(f"Saving model to path : {file_path}")
        model_data = {
            'W_in': self.W_in,
            'W_out': self.W_out,
            "losses": self.losses,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
        }

        with open(file_path, 'wb') as file:
            pickle.dump(model_data, file)

    def load_model(self, file_path):
        print(f"Load model from path : {file_path}")
        with open(file_path, 'rb') as file:
            model_data = pickle.load(file)
        self.W_in = model_data['W_in']
        self.W_out = model_data['W_out']
        self.losses = model_data["losses"]
        self.vocab = model_data["vocab"]
        self.vocab_size = model_data["vocab_size"]
        self.embedding_size = model_data["embedding_size"]

    def calcProcessTime(self, starttime, cur_iter, max_iter):

        telapsed = time.time() - starttime
        testimated = (telapsed / cur_iter) * (max_iter)

        finishtime = starttime + testimated
        finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

        lefttime = testimated - telapsed  # in seconds
        lefttime = dt.datetime.fromtimestamp(lefttime).strftime("%H:%M:%S")  # in time

        return ("%ss" % int(telapsed), "%ss" % lefttime, "%s" % finishtime)

    def visualize_tsne(self, num_words: int = 0, path=None, show=True):
        if num_words < 1:
            num_words = min(self.vocab_size, self.embedding_size)
        tsne = TSNE(n_components=2)
        word_vectors = self.W_in[:num_words]
        embedded_vectors = tsne.fit_transform(word_vectors)
        labels = []
        for index in range(num_words):
            labels.append(self.vocab[index])
        plt.figure(figsize=(30, 30))
        plt.scatter(embedded_vectors[:, 0], embedded_vectors[:, 1])
        for i, label in enumerate(labels):
            x, y = embedded_vectors[i, 0], embedded_vectors[i, 1]
            plt.annotate(label, (x, y), ha='center')
        if show:
            plt.show()
        if path:
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path)

    def export_word_embeddings(self, filepath):
        word_embeddings = self.W_in
        np.savetxt(filepath, word_embeddings, delimiter='\t')

    def export_metadata(self, filepath):
        metadata = pd.DataFrame({'Word': self.vocab})
        metadata.to_csv(filepath, header=False, sep='\t', index=False)

    def train(self, corpus, epochs, min_loss=float("-inf"), min_step=float("-inf")):
        print("")
        print(f"Target loss is {min_loss}")
        print("")
        self.build_vocab(corpus)
        self.initialize_weights()
        sen_len = len(corpus)
        prev_avg_loss = 10000
        for epoch in range(epochs):
            start = time.time()
            for si, sentence in enumerate(corpus):
                for i, center_word in enumerate(sentence):
                    context_words = self.get_context_window(sentence, i)
                    center_word_index = self.vocab.index(center_word)
                    context_words_indexes = [self.vocab.index(word) for word in context_words]
                    output_layer = self.forward_propagation(center_word_index, context_words_indexes)
                    self.backward_propagation(center_word_index, context_words_indexes, output_layer)
                    loss = self.calculate_loss(output_layer, center_word_index, context_words_indexes)
                    self.add_loss(loss)
                prstime = self.calcProcessTime(start, si + 1, sen_len + 1)
                sys.stdout.write(
                    f"\rEpoch: {epoch}/{epochs}\tElapsed: {prstime[0]} Left: {prstime[1]} Loss: {self.avg_loss()}\tSentence: {si}/{sen_len}\tFinish time {prstime[2]}")
            if self.avg_loss() < min_loss:
                print("")
                print(f"Exiting due to loss {self.avg_loss()} is less then min loss {min_loss}")
                print("")
                break
            if (prev_avg_loss - self.avg_loss()) <= 0 or (min_step > (prev_avg_loss - self.avg_loss())):
                print("")
                print(
                    f"Exiting step loss decreasing {prev_avg_loss - self.avg_loss()} is less than min step {min_step}")
                print("")
                break
            else:
                prev_avg_loss = self.avg_loss()

        sys.stdout.write(f"\rEpoch:{epochs}/{epochs}\tLoss:{self.avg_loss()}\tSentence:{sen_len}/{sen_len}\n")
        return self.losses


class SkipGram(Word2Vec):
    def __init__(self, window_size, embedding_size, learning_rate):
        super().__init__(window_size, embedding_size, learning_rate)

    def calculate_loss(self, output_probs, center_words, context_words):
        epsilon = 1e-8
        target_probs = output_probs[context_words]
        loss = -np.log(target_probs + epsilon).sum()
        return loss

    def forward_propagation(self, center_words, context_words):
        center_word_vector = self.W_in[center_words]
        hidden_layer = np.dot(center_word_vector, self.W_out)
        output_layer = self.softmax(hidden_layer)
        return output_layer

    def backward_propagation(self, center_words, context_words, output_layer):
        grad_hidden = output_layer.copy()
        grad_hidden[context_words] -= 1
        dW_out = np.outer(self.W_in[center_words], grad_hidden)
        dW_in = np.dot(grad_hidden, self.W_out.T)

        self.W_in[center_words] -= self.learning_rate * dW_in
        self.W_out -= self.learning_rate * dW_out

    def predict_context_words(self, center_word, top_k=5):
        center_word_index = self.vocab.index(center_word)
        hidden_layer = self.W_in[center_word_index]
        output_layer = np.dot(hidden_layer, self.W_out)
        sorted_indices = np.argsort(output_layer)[::-1]
        predicted_contexts = sorted_indices[:top_k]
        predicted_contexts = [self.vocab[word] for word in predicted_contexts]
        return predicted_contexts

    def cluster_words(self, num_clusters, path=None, show=False):
        tsne = TSNE(n_components=2)
        word_embeddings = self.W_in
        embedded_vectors = tsne.fit_transform(word_embeddings)

        kmeans = KMeans(n_clusters=num_clusters)
        cluster_labels = kmeans.fit_predict(embedded_vectors)

        # Visualize word clusters
        plt.figure(figsize=(12, 12))
        for i in range(num_clusters):
            cluster_points = embedded_vectors[cluster_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

        # Add word labels
        for i, label in enumerate(self.vocab):
            x, y = embedded_vectors[i, 0], embedded_vectors[i, 1]
            plt.annotate(label, (x, y), ha='center')
        if show:
            plt.legend()
            plt.show()
        if path:
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path)


class CBOW(Word2Vec):
    def __init__(self, window_size, embedding_size, learning_rate):
        super().__init__(window_size, embedding_size, learning_rate)

    def calculate_loss(self, output_probs, center_words, context_words):
        target_prob = output_probs[center_words]
        loss = -np.log(target_prob)
        return loss

    def forward_propagation(self, center_words, context_words):
        context_vectors = np.zeros((len(context_words), self.embedding_size))
        for i, word in enumerate(context_words):
            context_vectors[i] = self.W_in[word]
        hidden_layer = np.sum(context_vectors, axis=0)
        output_layer = self.softmax(np.dot(hidden_layer, self.W_out))
        return output_layer

    def backward_propagation(self, center_words, context_words, output_probs):
        grad_out = output_probs.copy()
        grad_out[center_words] -= 1
        grad_hidden = np.dot(grad_out, self.W_out.T)

        dW_out = np.outer(np.mean(self.W_in[context_words], axis=0), grad_out)
        dW_in = np.zeros_like(self.W_in)

        for word_index in context_words:
            dW_in[word_index] += np.mean(grad_hidden, axis=0)

        self.W_in -= self.learning_rate * dW_in
        self.W_out -= self.learning_rate * dW_out

    def predict_center_word(self, context_words):
        context_words = [self.vocab.index(word) for word in context_words]
        hidden_layer = np.mean(self.W_in[context_words], axis=0)
        output_layer = np.dot(hidden_layer, self.W_out)
        predicted_center = np.argmax(output_layer)
        return self.vocab[predicted_center]

    def cluster_words(self, num_clusters, path=None, show=False):
        tsne = TSNE(n_components=2)
        word_embeddings = self.W_in
        embedded_vectors = tsne.fit_transform(word_embeddings)

        kmeans = KMeans(n_clusters=num_clusters)
        cluster_labels = kmeans.fit_predict(embedded_vectors)

        # Visualize word clusters
        plt.figure(figsize=(200, 200))
        for i in range(num_clusters):
            cluster_points = embedded_vectors[cluster_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

        # Add word labels
        for i, label in enumerate(self.vocab):
            x, y = embedded_vectors[i, 0], embedded_vectors[i, 1]
            plt.annotate(label, (x, y), ha='center')
        if show:
            plt.legend()
            plt.show()
        if path:
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path)
