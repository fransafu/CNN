import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf

from TFRecord import TFRecord
from infrastructure.config import Config
from infrastructure import losses

def read_arguments():
    parser = argparse.ArgumentParser(description = "Entrena un modelo especificado")
    parser.add_argument("-config", type = str, help = "path to configuration file", required = True)
    parser.add_argument("-model", type=str, help=" name of model (resNet or others)", choices = ['resnet', 'alexnet'], required = True)              
    return parser.parse_args()

def load_dataset(config):
    # Generate
    tfr_train_file = os.path.join(config.dataset_dir, "train.tfrecords")
    tfr_test_file = os.path.join(config.dataset_dir, "test.tfrecords")

    if config.use_multithreads:
        tfr_train_file = [os.path.join(config.dataset_dir, "train_{}.tfrecords".format(idx)) for idx in range(config.num_threads)]
        tfr_test_file = [os.path.join(config.dataset_dir, "test_{}.tfrecords".format(idx)) for idx in range(config.num_threads)]        

    mean_file = os.path.join(config.dataset_dir, "mean.dat")
    shape_file = os.path.join(config.dataset_dir, "shape.dat")

    return tfr_train_file, tfr_test_file, mean_file, shape_file

def main():
    pargs = read_arguments()
    model_name = pargs.model

    config = Config()
    config.load(Path(pargs.config))

    # Dataset: Load
    _, tfr_test_file, mean_file, shape_file = load_dataset(config)

    input_shape = np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)

    number_of_classes = config.num_classes

    val_dataset = tf.data.TFRecordDataset(tfr_test_file)
    val_dataset = val_dataset.map(lambda x : TFRecord.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation = False));    
    val_dataset = val_dataset.batch(batch_size = config.batch_size)

    # This code allows program to run in  multiple GPUs. It was tested with 2 gpus.
    tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        load_from = os.path.join(config.dataset_dir, f"{model_name}_model")

        model = tf.keras.models.load_model(load_from, custom_objects={'crossentropy_loss': losses.crossentropy_loss})

        model.summary()

        model.evaluate(val_dataset, steps = config.validation_steps, callbacks=[tensorboard_callback])

if __name__ == "__main__":
    main()
