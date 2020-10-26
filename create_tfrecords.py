import argparse

from pathlib import Path
from TFRecord import TFRecord
from Dataset import Dataset
from Config import Config

def read_arguments():
    parser = argparse.ArgumentParser(description = "Entrena un modelo especificado")
    parser.add_argument("-config", type = str, help = "path to configuration file", required = True)
    parser.add_argument("-model", type=str, help=" name of model (resNet or others)", choices = ['resnet', 'alexnet'], required = True)              
    return parser.parse_args()

def create_tfrecords(config):
    dataset = Dataset(config)
    dataset.create_tfrecords('train', config.image_type)
    dataset.create_tfrecords('test', config.image_type)

def main():
    pargs = read_arguments()

    config = Config()
    config.load(Path(pargs.config))
    # Dataset: Generate tfrecords
    # create_tfrecords(config)

if __name__ == "__main__":
    main()
