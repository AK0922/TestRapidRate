from gensim.models.doc2vec import Doc2Vec
from utils.options import Options

from utils import log_helper

log = log_helper.get_logger("Doc2Vec_Helper")


def init_doc2vec_model(tagged_reviews):

    model = Doc2Vec(min_count=25, iter=50, workers=6, size=1000)
    model.build_vocab(tagged_reviews)

    return model


def train_doc2vec_model(doc2vec_model, tagged_reviews):

    shuffle_count = Options.doc2vec_training_count

    for i in range(shuffle_count):
        log.info("Shuffles left: " + str(shuffle_count - i))
        doc2vec_model.train(tagged_reviews)
