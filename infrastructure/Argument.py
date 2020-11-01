import argparse

class Argument:
    def read_arguments(self):
        parser = argparse.ArgumentParser(description = "Entrena un modelo especificado")
        parser.add_argument("-config", type = str, help = "path to configuration file", required = True)
        parser.add_argument("-model", type=str, help=" name of model (resNet or others)", choices = ['resnet', 'alexnet'], required = True)              
        return parser.parse_args()
