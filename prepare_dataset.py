from pathlib import Path

from infrastructure.Argument import Argument
from infrastructure.Dataset import Dataset
from infrastructure.config import Config

def main():
    args = Argument()
    pargs = args.read_arguments()

    config = Config()
    config.load(Path(pargs.config))
    config.isValid()

    # Dataset: Generate tfrecords
    dataset = Dataset(config)
    dataset.create_tfrecords('train', config.image_type)
    dataset.create_tfrecords('test', config.image_type)

if __name__ == "__main__":
    main()
