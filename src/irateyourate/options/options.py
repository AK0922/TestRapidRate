from argparse import ArgumentParser

from src.irateyourate.enums.vectorization_tools import VectorizationTools


class Options(object):
    input_file_path = None
    vectorization_tool = None
    output_file_path = None


def parse_args(argv):

    parser = ArgumentParser(prog="IRateYouRate")
    parser.add_argument('--input_file_path', metavar='Input File Path', type=str)
    parser.add_argument('--vectorization_tool', metavar='Vectorization Tool', type=VectorizationTools)
    parser.add_argument('--output_file_path', metavar='Output File Path', type=str)

    options = parser.parse_args(argv, namespace=Options)

    return options
