from src.train.trainer import Word2VecTrainer


def build_file_name(current_model, epoch, window, dim, ds):
    return f"{current_model}__epoch-{epoch}__window-{window}__dim-{dim}__ds-{ds}"


if __name__ == '__main__':
    # Input sentences

    num_epochs = [1, 2, 5]
    datasets = [("data/tiny_dataset.txt", "tiny")]
    # datasets = [("data/tiny_dataset.txt", "tiny"), ("data/bigger_dataset.txt", "bigger"), ]
    embedding_dimensions = [500]
    windows = [2, 5]
    models = ["skip-gram", "cbow"]

    trainer = Word2VecTrainer(num_epochs=num_epochs,
                              datasets=datasets,
                              embedding_dimensions=embedding_dimensions,
                              windows=windows,
                              models=models)

    trainer.train_word2vec(models_target={"skip-gram": 45, "cbow": 35})