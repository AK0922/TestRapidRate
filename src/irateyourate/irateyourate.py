import sys

from src.irateyourate.processors.amazon_review_processor import AmazonReviewProcessor
from src.irateyourate.utils.options import Options
from argparse import ArgumentParser


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
    parser = ArgumentParser(prog="IRateYouRate")
    parser.add_argument('--input_file_path', metavar='Input File Path', type=str)
    parser.add_argument('--doc2vec_training_count', metavar='Docvec Training Count', type=int)
    parser.add_argument('--doc2vec_model_path', metavar='Docvec Model Path', type=str)
    parser.add_argument('--ml_model_path', metavar='ML Model Path', type=str)

    Options.options = parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
