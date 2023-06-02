from src.model.Word2Vec import SkipGram, CBOW
from src.utils.data_io import list_files


def one_hot_encode(vocab_size, indexes):
    if type(indexes) is int:
        indexes = [indexes]
    one_hot = [0] * vocab_size
    for x in indexes:
        one_hot[x] = 1
    return one_hot


def one_hot_encode_array(vocab_size, indexes):
    one_hot = [0] * vocab_size
    for x in indexes:
        one_hot[x] = 1
    return one_hot


if __name__ == '__main__':
    for f in list_files("./models_trained"):
        model = None
        if "bigger" in f:
            if f.startswith("cbow"):
                model = SkipGram(window_size=0, embedding_size=0, learning_rate=0.001)
            else:
                model = CBOW(window_size=0, embedding_size=0, learning_rate=0.001)
            model.load_model("./models_trained/" + f)
            fname = f.replace(".pkl", "")
            loss = str(model.avg_loss()).replace(".", "_")
            model.export_word_embeddings("./models_metadata/" + fname + "_loss_" + loss  + "_embedings.tsv")
            model.export_metadata("./models_metadata/" + fname + "_loss_" + loss  + "_metadat.tsv")
            # model.cluster_words(5, "./models_visualizations/" + fname + "_loss_" + loss + '_cluster.png')
