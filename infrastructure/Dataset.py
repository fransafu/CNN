import os
import re

import numpy as np
import tensorflow as tf

from infrastructure.TFRecord import TFRecord
import infrastructure.imgproc as imgproc

class Dataset:
    def __init__(self, config) -> None:
        self.dataset_dir = f"{config.basepath_dir}/dataset"
        self.config = config

    def create_tfrecords(self, type_dataset=None, image_type=None, process_fun=imgproc.process_image):
        if image_type == 'SKETCH': 
            process_fun = imgproc.process_sketch        
        elif image_type == 'MNIST':
            process_fun = imgproc.process_mnist
        TFRecord.create(self.config, type_dataset, processFun=process_fun)

    def load(self):
        tfr_train_file, tfr_test_file, mean_file, shape_file = self.__get_files()
        input_shape = np.fromfile(shape_file, dtype=np.int32)
        mean_image = np.fromfile(mean_file, dtype=np.float32)
        mean_image = np.reshape(mean_image, input_shape)

        # Loading tfrecords into dataset object
        tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
        tr_dataset = tr_dataset.map(lambda x : TFRecord.parser_tfrecord(x, input_shape, mean_image, self.config.num_classes, with_augmentation = True));    
        tr_dataset = tr_dataset.shuffle(self.config.shuffle_size)
        tr_dataset = tr_dataset.batch(batch_size = self.config.batch_size)

        val_dataset = tf.data.TFRecordDataset(tfr_test_file)
        val_dataset = val_dataset.map(lambda x : TFRecord.parser_tfrecord(x, input_shape, mean_image, self.config.num_classes, with_augmentation = False));    
        val_dataset = val_dataset.batch(batch_size = self.config.batch_size)

        return input_shape, tr_dataset, val_dataset

    def __get_files(self):
        tfr_train_file = []
        tfr_test_file = []
        for file in os.listdir(self.dataset_dir):
            if re.findall("train.*.tfrecords", file):
                tfr_train_file.append(os.path.join(self.dataset_dir, file))
            elif re.findall("test.*.tfrecords", file):
                tfr_test_file.append(os.path.join(self.dataset_dir, file))
            mean_file = os.path.join(self.dataset_dir, "mean.dat")
            shape_file = os.path.join(self.dataset_dir, "shape.dat")
        return tfr_train_file, tfr_test_file, mean_file, shape_file
