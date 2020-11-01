import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import infrastructure.losses
import numpy as np
import os
from infrastructure.TFRecord import TFRecord

def display_training_curves(training, validation, title, subplot, savefile=False, filename=''):
  ax = plt.subplot(subplot)
  ax.plot(training)
  ax.plot(validation)
  ax.set_title('model '+ title)
  ax.set_ylabel(title)
  ax.set_xlabel('epoch')
  ax.legend(['training', 'validation'])
  if savefile:
      ax.figure.savefig(filename)

# Load model
model = tf.keras.models.load_model('./experimentos/resnet_pretrain_50_epoch/30_10_2020__02_29_36_resnet_model/', custom_objects={ 'crossentropy_loss': losses.crossentropy_loss })

# Load history
history = pickle.load(open('./experimentos/resnet_pretrain_50_epoch/30_10_2020__02_29_36_resnet_history', "rb"))

# Show summary
model.summary()
""" 
plt.subplots(figsize=(10,10))
plt.tight_layout()

display_training_curves(history['accuracy'], history['val_accuracy'], 'accuracy', 211)
display_training_curves(history['loss'], history['val_loss'], 'loss', 212, True, 'plot.png')
 """



dataset_path = './dataset'
model_path = './experimentos/resnet_pretrain_50_epoch/30_10_2020__02_29_36_resnet_model/'
history_path = './experimentos/resnet_pretrain_50_epoch/30_10_2020__02_29_36_resnet_history'
use_multithreads = True
num_threads = 8
number_of_classes = 19

def get_mapping():
  classes = {}
  with open('mapping.txt', 'r') as mapping_file:
    for clas in mapping_file.readlines():
      clas_map = clas.replace('\n', '').split('\t')
      if len(clas_map) > 0:
        classes[clas_map[1]] = clas_map[0]
  return classes

def load_dataset(dataset_path, use_multithreads, num_threads):
    # Generate
    tfr_train_file = os.path.join(dataset_path, "train.tfrecords")
    tfr_test_file = os.path.join(dataset_path, "test.tfrecords")

    if use_multithreads:
        tfr_train_file = [os.path.join(dataset_path, "train_{}.tfrecords".format(idx)) for idx in range(num_threads)]
        tfr_test_file = [os.path.join(dataset_path, "test_{}.tfrecords".format(idx)) for idx in range(num_threads)]        

    mean_file = os.path.join(dataset_path, "mean.dat")
    shape_file = os.path.join(dataset_path, "shape.dat")

    return tfr_train_file, tfr_test_file, mean_file, shape_file

_, tfr_test_file, mean_file, shape_file = load_dataset(dataset_path, use_multithreads, num_threads)

input_shape = np.fromfile(shape_file, dtype=np.int32)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

val_dataset = tf.data.TFRecordDataset(tfr_test_file)
val_dataset = val_dataset.map(lambda x : TFRecord.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation = False))

val_dataset = list(val_dataset)

classes = get_mapping()

print(classes)



""" for i in range(len(val_dataset)):
  img = val_dataset[i][0]
  val_clas = np.argmax(val_dataset[i][1])

  # preprocess image
  im = 255 * (img - np.min(img))/ (np.max(img)-np.min(img))
  
  # predict
  pred = model.predict(tf.expand_dims(img, 0))

  # get class predicted
  pred_clas = np.argmax(pred[0])
 """

"""
fig, axs = plt.subplots(h, w, figsize=(1.6*h,1.6*w))
fig.subplots_adjust(bottom=0.5)
for i in range(h):
  for j in range(w):

    idx_sample = np.random.randint(len(val_dataset) - 1)
    img = val_dataset[idx_sample][0]

    im = 255 * (img - np.min(img))/ (np.max(img)-np.min(img))

    val_clas = np.argmax(val_dataset[idx_sample][1])
    pred = model.predict(tf.expand_dims(img, 0))
    pred_clas = np.argmax(pred[0])

    axs[i,j].set_title(f"class: {classes.get(str(val_clas))} \n pred: {classes.get(str(pred_clas))}")
    axs[i,j].grid(False)
    axs[i,j].set_xticklabels([])
    axs[i,j].set_yticklabels([])
    axs[i,j].imshow(np.uint8(im))
"""