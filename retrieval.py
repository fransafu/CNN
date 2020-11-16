import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd

from infrastructure.Argument import Argument
from infrastructure.config import Config
from infrastructure.Image import Image
import infrastructure.imgproc as imgproc
from infrastructure import losses
from infrastructure.errors import ModelNotSelected

from models.alexnet_embedding import AlexNetEmbedding
from models.resnet_embedding import ResNetEmbedding


def buscar_cercanos(embedding, dataset):
    dist = np.sqrt(np.sum(np.square(dataset - embedding), axis=1))
    sorted_idx = sorted(range(len(dataset)), key=lambda x: dist[x])
    return sorted_idx

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

    # Load de dataset
    shape_file = os.path.join(config.dataset_dir, 'shape.dat')
    input_shape = np.fromfile(shape_file, dtype=np.int32)

    mean_file = os.path.join(config.dataset_dir, 'mean.dat')
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)

    datafile = os.path.join(config.dataset_dir, 'train.txt')
    with open(datafile) as file:
        lines = [line.rstrip() for line in file]
        _lines = [tuple(line.rstrip().split('\t')) for line in lines]
        train_filenames, train_labels = zip(*_lines)

    datafile = os.path.join(config.dataset_dir, 'test.txt')
    with open(datafile) as file:
        lines = [line.rstrip() for line in file]
        _lines = [tuple(line.rstrip().split('\t')) for line in lines]
        test_filenames, test_labels = zip(*_lines)

    # Select the model
    model = None
    if model_name == 'alexnet':
        model = AlexNetEmbedding(config.num_classes)
    elif model_name == 'resnet':
        model = ResNetEmbedding([3, 4, 6, 3], [64, 128, 256, 512], config.num_classes, se_factor=0)

    # Validate if select a model
    if model is None:
        raise ModelNotSelected()

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
    if config.pretrained_weights: # Check if not empty in the config file
        model.load_weights(config.pretrained_weights, by_name=True, skip_mismatch=True)

    # Load the pre-generated embeddings
    embeddings_dataset = np.fromfile("embeddings_{}.npy".format(model_name),dtype=np.float32).reshape([-1,1024])

    result = []
    for i in range(len(test_filenames)):

        # The filename to test
        filename = test_filenames[i]

        # Validate if the datafile exists
        if not os.path.exists(datafile):
            break

        # Load the image and preprocess
        target_size = (config.image_height,config.image_width)
        image = imgproc.process_image(Image.read_image('dataset/' + filename, config.channels), target_size)
        image = image - mean_image
        image = tf.expand_dims(image, 0)

        # Predict the class
        embedding = model.predict(image)

        # Looking for a neighbor candidates
        candidatos = buscar_cercanos(embedding, embeddings_dataset)

        # Store the results (class_target, top10 predicts, labels)
        class_to_search = int(test_labels[i])
        top10_idx_mejores = candidatos[:10]
        top10_labels_mejores = [train_labels[k] for k in candidatos[:10]]
        result.append({
            'class_to_search': class_to_search,
            'filename': filename,
            'top10_search': '|'.join([str(k) for k in top10_labels_mejores]),
            'top10_idx_mejores': '|'.join([str(k) for k in top10_idx_mejores])
        })

    # Save the results into csv file
    df = pd.DataFrame(result)
    df.to_csv(f'results_{model_name}.csv')

if __name__ == '__main__':
    main()
