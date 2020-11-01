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

    def __write_history(self, history) -> str:
        filename_history = f"{self.timestamp}_{self.config.model_name}_history"
        with open(filename_history, 'wb') as history_file:
            pickle.dump(history.history, history_file)
        return filename_history

    def experiment(self, history, model_pathname, config_filename) -> None:
        name_zipfile = f"{self.timestamp}.zip"

        history_filename = self.__write_history(history)

        zipfile = Zip(self.basepath_experiment, name_zipfile)
        zipfile.zipdir(model_pathname)
        zipfile.zipfile(history_filename)
        zipfile.zipfile(config_filename)
