import os
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import pickle

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

        print("Guardando experimento")
        print(f"{self.basepath_experiment}/{name_zipfile}")
        with ZipFile(f"{self.basepath_experiment}/{name_zipfile}", 'w', ZIP_DEFLATED) as zipfile:
            for root, _, files in os.walk(model_filename):
                for file in files:
                    zipfile.write(os.path.join(root, file))
            zipfile.write(history_filename)
            zipfile.write(config_filepath)
