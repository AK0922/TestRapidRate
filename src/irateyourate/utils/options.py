from argparse import ArgumentParser


class Options(object):
    input_file_path = None
    vectorization_tool = None
    doc2vec_model_path = None
    doc2vec_training_count = None
    ml_model_path = None
    options = None


def parse_args(argv):

    parser = ArgumentParser(prog="IRateYouRate")
    parser.add_argument('--input_file_path', metavar='Input File Path', type=str)
    parser.add_argument('--vectorization_tool', metavar='Vectorization Tool', type=str)
    parser.add_argument('--doc2vec_training_count', metavar='Docvec Training Count', type=int)
    parser.add_argument('--doc2vec_model_path', metavar='Docvec Model Path', type=str)
    parser.add_argument('--ml_model_path', metavar='ML Model Path', type=str)

    Options.options = parser.parse_args(argv, namespace=Options)
