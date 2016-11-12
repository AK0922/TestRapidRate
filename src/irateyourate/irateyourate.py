import sys

from src.irateyourate.processors.amazon_review_processor import AmazonReviewProcessor
from src.irateyourate.utils import options


def main(argv):
    options.parse_args(argv)
    processor = AmazonReviewProcessor()
    processor.process()


if __name__ == "__main__":
    main(sys.argv[1:])
