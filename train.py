from pathlib import Path

import tensorflow as tf

from infrastructure.config import Config
from infrastructure import losses
from infrastructure.Save import Save
from infrastructure.Argument import Argument
from infrastructure.Dataset import Dataset

import models.resnet as resnet
from models.alexnet import AlexNetModel

def main():
    args = Argument()
    pargs = args.read_arguments()

    model_name = pargs.model

    config = Config()
    config.load(Path(pargs.config))
    config.isValid()

    # DATASET SECTION
    dataset = Dataset(config)
    input_shape, tr_dataset, val_dataset = dataset.load()

    # TF SESSION SECTION
    # This code allows program to run in  multiple GPUs. It was tested with 2 gpus.
    tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        # MODEL SELECTION SECTION
        model = None
        if model_name == 'alexnet':
            model = AlexNetModel(config.num_classes)
        else:
            model = resnet.ResNet([3,4,6,3], [64,128,256,512], config.num_classes, se_factor = 0)

        input_image = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_image')     
        model(input_image)
        model.summary()

        if config.use_imagenet_weights:
            model.load_weights(config.imagenet_dir, by_name=True, skip_mismatch=True)

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
                        validation_steps = float(config.validation_steps)
        )

        # Save history
        model_save_to = f"{model_name}_model"
        model.save(model_save_to)

        save = Save(config)
        save.experiment(history, model_save_to, pargs.config)

if __name__ == "__main__":
    main()
