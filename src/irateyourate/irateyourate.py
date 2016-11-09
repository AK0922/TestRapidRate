import sys
from src.irateyourate.options import options


def main(argv):
    opts = options.parse_args(argv)
    print(opts.input_file_path)


if __name__ == "__main__":
    main(sys.argv[1:])
