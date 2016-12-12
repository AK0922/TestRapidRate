import sys
from argparse import ArgumentParser

from processors.amazon_review_processor import AmazonReviewProcessor
from utils.options import Options


def main(argv):
    """
    Main function to kick start execution
    :param argv:
    :return: null
    """
    parse_args(argv)
    processor = AmazonReviewProcessor()
    processor.process()


def parse_args(argv):
    """
    Parses command line arguments form an options object
    :param argv:
    :return:
    """
    parser = ArgumentParser(prog="rapid-rate")
    parser.add_argument('--input_file_path', metavar='Input File Path', type=str)
    parser.add_argument('--doc2vec_training_count', metavar='Docvec Training Count', type=int)
    parser.add_argument('--doc2vec_model_path', metavar='Docvec Model Path', type=str)
    parser.add_argument('--ml_model_path', metavar='ML Model Path', type=str)

    Options.options = parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
