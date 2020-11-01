import os
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import pickle

class Zip:
    def __init__(self, basepath_store, zip_filename) -> None:
        self.zipfile = ZipFile(f"{basepath_store}/{zip_filename}", 'w', ZIP_DEFLATED)

    def zipfile(self, filepath) -> None:
        self.zipfile.write(filepath)

    def zipdir(self, path) -> None:
        for root, _, files in os.walk(path):
            for file in files:
                self.zipfile.write(os.path.join(root, file))

class Save:
    def __init__(self, config) -> None:
        self.config = config
        self.basepath_experiment = f"{self.config.basepath_dir}/experiments"
        self.timestamp = datetime.now().strftime('%d%m%Y%H%M%S')

    def __save_history(self, history, model_filename) -> str:
        filename_history = f"{self.timestamp}_{model_filename}_history"
        with open(filename_history, 'wb') as history_file:
            pickle.dump(history.history, history_file)
        return filename_history

    def experiment(self, history, model_filename, config_filepath) -> None:
        name_zipfile = f"{self.timestamp}_{model_filename}.zip"

        history_filename = self.__save_history(history, model_filename)

        zipfile = Zip(self.basepath_experiment, name_zipfile)
        zipfile.zipdir(model_filename)
        zipfile.zipfile(history_filename)
        zipfile.zipfile(config_filepath)
        zipfile.close()
