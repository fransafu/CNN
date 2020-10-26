import os
import random


class File:
    @staticmethod
    def validate_labels(labels):
        """
            It checks if labels are in the correct format [int]
            labels need to be integers, from 0 to NCLASSES -1
        """
        new_labels = [int(label) for label in labels]
        label_set = set(new_labels)
        #checking the completness of the label set
        if (len(label_set) == max(label_set) + 1) and (min(label_set) == 0):
            return new_labels
        else:
            raise ValueError("Some codes are missed in label set! {}".format(label_set))


    @staticmethod
    def read_data_from_file(file_path, dataset = "train" , shuf = True):
        """
            Read data from text files and apply shuffle by default
        """
        datafile = os.path.join(file_path, dataset + ".txt")
        assert os.path.exists(datafile)
        # reading data from files, line by line
        with open(datafile) as file:
            lines = [line.rstrip() for line in file]
            if shuf:
                random.shuffle(lines)
            _lines = [tuple(line.rstrip().split('\t'))  for line in lines ]
            filenames, labels = zip(*_lines)
            labels = File.validate_labels(labels)
        return filenames, labels
