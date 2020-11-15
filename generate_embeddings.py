import os
from pathlib import Path
import tensorflow as tf
import numpy as np

from infrastructure.Argument import Argument
from infrastructure.config import Config
from infrastructure.Dataset import Dataset
from infrastructure.Image import Image
import infrastructure.imgproc as imgproc
from infrastructure import losses
from infrastructure.errors import ModelNotSelected

from models.alexnet_embedding import AlexNetEmbedding
from models.resnet_embedding import ResNetEmbedding


def main():
    # Load the CLI arguments
    args = Argument()
    pargs = args.read_arguments()

    # The model name used
    model_name = pargs.model

    # Load the config file and validate if the basepath is the same at declaring in the config file (You need to check it)
    config = Config()
    config.load(Path(pargs.config))
    config.isValid()

    # Load de dataset from config path (you need to put the dataset data into dataset folder)
    dataset = Dataset(config)
    input_shape, tr_dataset, val_dataset, shape_file, mean_image = dataset.load()

    # Select the model
    model = None
    if model_name == 'alexnet':
        model = AlexNetEmbedding(config.num_classes)
    elif model_name == 'resnet':
        model = ResNetEmbedding([3, 4, 6, 3], [64, 128, 256, 512], config.num_classes, se_factor=0)

    # Validate if select a model
    if model is None:
        raise ModelNotSelected()

    # Load filenames and labels from train dataset
    datafile = os.path.join(config.dataset_dir, 'train.txt')
    with open(datafile) as file:
        lines = [line.rstrip() for line in file]
        _lines = [tuple(line.rstrip().split('\t')) for line in lines]
        filenames, labels = zip(*_lines)

    # Config the model 'build' option (used the input_shape)
    model.build((1, input_shape[0], input_shape[1], input_shape[2]))

    # Print the model summary
    model.summary()

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

    # Compile the model
    model.compile(optimizer=opt, loss=losses.crossentropy_loss)

    # Load pre-trained weights
    if config.weights_path: # Check if not empty in the config file
        model.load_weights(config.pretrained_weights, by_name=True, skip_mismatch=True)

    embeddings = []
    target_size = (config.image_height, config.image_width)
    for i, filename in enumerate(filenames):
        # Load the image
        image = imgproc.process_image(Image .read_image('dataset/' + filename, config.channels), target_size)
        image = image - mean_image
        image = tf.expand_dims(image, 0)
        # Get the predict and save as "embedding"
        embedding = model.predict(image)[0]
        embeddings.append(embedding)

    # Save the embeddings with the model name (careful if you have an embeddings file with the same name, your file will be rewritten)
    np.asarray(embeddings).tofile("embeddings_{}.npy".format(pargs.model))

if __name__ == '__main__':
    main()
