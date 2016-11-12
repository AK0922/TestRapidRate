from src.irateyourate.utils import log_helper
from src.irateyourate.utils.options import Options
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import sent_tokenize, word_tokenize
from random import shuffle

log = log_helper.get_logger("Doc2Vec_Helper")


def init_doc2vec_model(tagged_reviews):

    model = Doc2Vec(min_count=2, size=500, iter=20, workers=1)
    model.build_vocab(tagged_reviews)

    return model


def train_doc2vec_model(doc2vec_model, tagged_reviews):

    shuffle_count = Options.doc2vec_training_count

    for i in range(shuffle_count):
        log.info("Shuffles left: " + str(shuffle_count - i))
        doc2vec_model.train(tagged_reviews)
