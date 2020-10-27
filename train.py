import argparse
import os
from pathlib import Path
import pickle
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import tensorflow as tf

from TFRecord import TFRecord
from Config import Config
import losses

import models.resnet as resnet
from models.alexnet import AlexNetModel


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

"""
def find_latest_ckp(ckp_path):
    latest_ckp = None
    for file in os.listdir(ckp_path):
        if file.endswith(".h5"):
            new_version = int(file.split('.h5')[0])
            if latest_ckp:
                if latest_ckp <= new_version:
                    latest_ckp = new_version
            else:
                latest_ckp = new_version
    return latest_ckp
"""

def save_history(history_filename, history):
    with open(history_filename, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def consolidate_experiment(now, model_path, history_path, config_path):
    name_zipfile = f"{now.strftime('%d_%m_%Y__%H_%M_%S')}.zip"

    with ZipFile(name_zipfile, 'w') as myzip:
        myzip.write(model_path)
        myzip.write(history_path)
        myzip.write(config_path)


def main():
    pargs = read_arguments()

    now = datetime.now()

    model_name = pargs.model

    config = Config()
    config.load(Path(pargs.config))

    # DATASET SECTION
    tfr_train_file, tfr_test_file, mean_file, shape_file = load_dataset(config)

    input_shape = np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)

    number_of_classes = config.num_classes

    # Loading tfrecords into dataset object
    tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
    tr_dataset = tr_dataset.map(lambda x : TFRecord.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation = True));    
    tr_dataset = tr_dataset.shuffle(config.shuffle_size)
    tr_dataset = tr_dataset.batch(batch_size = config.batch_size)

    val_dataset = tf.data.TFRecordDataset(tfr_test_file)
    val_dataset = val_dataset.map(lambda x : TFRecord.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation = False));    
    val_dataset = val_dataset.batch(batch_size = config.batch_size)

    # TF SESSION SECTION
    # This code allows program to run in  multiple GPUs. It was tested with 2 gpus.
    tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # CALLBACK SECTION

        # checkpoint_path = "cp-{epoch:04d}.ckpt"
        # checkpoint_dir = f"{config.snapshot_dir}/{checkpoint_path}"

        # callback
        # logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        # Defining callback for saving checkpoints
        # save_freq: frecuency in terms of number steps each time checkpoint is saved 

        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_dir, # config.snapshot_dir + '{epoch:03d}.h5',
        #     save_weights_only=True,
        #     verbose=1,
        #     mode = 'max',
        #     monitor='val_acc',
        #     save_freq = 'epoch',            
        # )

        # MODEL SELECTION SECTION
        model = None
        if model_name == 'alexnet':
            model = AlexNetModel(config.num_classes)
        else:
            # Resnet 34
            model = resnet.ResNet([3,4,6,3], [64,128,256,512], config.num_classes, se_factor = 0)

        print(f"""

            Input image, shape: ({input_shape[0]}, {input_shape[1]}, {input_shape[2]})
            Model: {model_name}

        """)

        input_image = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_image')     
        model(input_image)
        model.summary()

        # CHECKPOINT SECTION
        # Use_checkpoints to load weights
        # if config.use_checkpoint and find_latest_ckp(config.snapshot_dir):
        #     #latest = tf.train.latest_checkpoint(config.snapshot_dir)
        #     #print(latest)
        #     print(config.use_checkpoint)
        #     print(find_latest_ckp(config.snapshot_dir))
        #     ckp_filepath = config.snapshot_dir + '{epoch:03d}.h5'.format(epoch=find_latest_ckp(config.snapshot_dir))
        #     print(ckp_filepath)
        #     model.load_weights(ckp_filepath, by_name = True, skip_mismatch = True)
        # else:
        #     model.save_weights(checkpoint_path.format(epoch=0))

        # OPTIMIZER SECTION
        opt = None
        if config.optimizer_name == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)
        elif config.optimizer_name == 'sgd':
            # Definin optimizer, my experince shows that SGD + cosine decay is a good starting point        
            # Recommended learning_rate is 0.1, and decay_steps = total_number_of_steps                        
            lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate = config.learning_rate,
                                                            decay_steps = config.decay_steps,
                                                            alpha = 0.0001)
            opt = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum = 0.9, nesterov = True)        

        model.compile(optimizer=opt, loss=losses.crossentropy_loss, metrics=['accuracy'])        

        history = model.fit(tr_dataset, 
                        epochs = config.num_epochs,
                        validation_data=val_dataset,
                        validation_steps = float(config.validation_steps)# ,
                        # callbacks=[model_checkpoint_callback, tensorboard_callback]
        )

        # Save history
        save_history_to = f"{now.strftime('%d_%m_%Y__%H_%M_%S')}_{model_name}_history"
        save_history(save_history_to, history)

        # Save de model
        save_model_to = f"{now.strftime('%d_%m_%Y__%H_%M_%S')}_{model_name}_model"
        model.save(save_model_to)
        print("model saved")

        consolidate_experiment(now, save_model_to, save_history_to, pargs.config)

if __name__ == "__main__":
    main()
