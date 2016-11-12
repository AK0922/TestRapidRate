import json
from src.irateyourate.utils import log_helper
from src.irateyourate.utils.options import Options
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import sent_tokenize, word_tokenize

log = log_helper.get_logger("ReviewFile_Helper")


def parse_review_file():
    """
    Parses the input review file
    :return: a list of TaggedDocs for Doc2Vec and a dict of scores
    """

    tagged_reviews = list()
    rating_dict = dict()
    for review in open(Options.options.input_file_path):
        identifier, tagged_review, rating = parse_review(json.loads(review))

        tagged_reviews.append(tagged_review)
        rating_dict[identifier] = rating

    return tagged_reviews, rating_dict


def parse_review(review):
    """
    :param review: JSON object containing an Amazon review
    :return: Review Identifier, TaggedDocument for Doc2Vec usage, Review Rating
    """

    identifier = review['reviewerID'] + review['asin']
    rating = review['overall']

    review_text = review['reviewText']

    sentence_tokens = sent_tokenize(review_text)
    all_words = list()
    for sentence_token in sentence_tokens:
        all_words.extend(word_tokenize(sentence_token))

    tagged_review = TaggedDocument(words=all_words, tags=[identifier])

    return identifier, tagged_review, rating


