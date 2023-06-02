import os
from typing import Dict

from src.model.Word2Vec import SkipGram, CBOW
from src.preprocess.preprocess import preprocess_dataset
from src.utils.data_io import load_data
from src.utils.text_utils import build_corpus


class Word2VecTrainer:

    def __init__(self,
                 num_epochs, datasets, embedding_dimensions, windows,
                 models
                 ):
        self.num_epochs = num_epochs
        self.datasets = datasets
        self.embedding_dimensions = embedding_dimensions
        self.windows = windows
        self.models = models
        self.iterations = len(num_epochs) * len(datasets) * len(embedding_dimensions) * len(windows) * len(models)
        self.visualizations_prefix = "./models_visualizations"
        self.models_prefix = "./models_trained"

    def build_file_name(self, current_model, epoch, window, dim, ds):
        return f"{current_model}__epoch-{epoch}__window-{window}__dim-{dim}__ds-{ds}"

    def train_word2vec(self, models_target: Dict[str, int]):
        target_loss = 100
        cur_iteration = 0
        for ds in self.datasets:
            dataset = load_data(ds[0])
            dataset = preprocess_dataset(dataset)
            dataset = build_corpus(dataset)
            for epoch in self.num_epochs:

                for dim in self.embedding_dimensions:
                    for window in self.windows:
                        for current_model in self.models:
                            target_loss = min(models_target[current_model], target_loss)
                            is_loaded = False
                            print(
                                f"Iteration {cur_iteration}/{self.iterations} |  Model: {current_model} | Epoch: {epoch}   | Window {window} | Dimensions {dim} | Dataset {ds[1]}")
                            model = None
                            if current_model == "skip-gram":
                                model = SkipGram(window_size=window, embedding_size=dim, learning_rate=0.001)
                            else:
                                model = CBOW(window_size=window, embedding_size=dim, learning_rate=0.001)
                            cur_iteration += 1
                            if os.path.exists(
                                    f"{self.models_prefix}/{self.build_file_name(current_model, epoch, window, dim, ds[1])}.pkl"):
                                print("Loading model due to existence...")
                                is_loaded = True
                                model.load_model(
                                    f"{self.models_prefix}/{self.build_file_name(current_model, epoch, window, dim, ds[1])}.pkl")
                            else:
                                model.train(dataset, epoch, min_loss=target_loss * 0.4, min_step=0.001)

                            if current_model == "cbow":
                                print(
                                    f"Center word for [theatre, movie] is:  {model.predict_center_word(['theatre', 'movie'])}")
                            else:
                                print(f"Context words for 'theatre' are: {model.predict_context_words('theatre')}")
                            print(f"Loss: {model.avg_loss()}")

                            if model.avg_loss() < target_loss:
                                if not is_loaded:
                                    model.save_model(
                                        f"{self.models_prefix}/{self.build_file_name(current_model, epoch, window, dim, ds[1])}.pkl")
                                    model.visualize_tsne(show=False,
                                                         path=f"{self.visualizations_prefix}/{self.build_file_name(current_model, epoch, window, dim, ds[1])}.png")
                                else:
                                    model.visualize_tsne(show=True)

                            print(
                                f"======================================================================================================")

        print("Training completed.")
