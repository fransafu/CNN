import os
from infrastructure.File import File
import sys
import numpy as np
import tensorflow as tf
import threading
from datetime import datetime

from infrastructure.Image import Image
import infrastructure.imgproc as imgproc


class TFRecord:
    """ TFRecord generator """

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def create_from_file(data_dir, filenames, labels, image_shape, tfr_filename, process_function = imgproc.resize_image):    
        # Create tf-records
        writer = tf.io.TFRecordWriter(tfr_filename)
        print("tfr_filename: ", tfr_filename)
        # Filenames and labels should have the same size
        assert len(filenames) == len(labels)
        mean_image = np.zeros(image_shape, dtype=np.float32)
        n_reading_error = 0    
        for i in range(len(filenames)):
            try :
                if i % 500 == 0 or (i + 1) == len(filenames):
                    print("---{}".format(i))
                image = Image.read_image(f"{data_dir}/dataset/{filenames[i]}", image_shape[2]) #scikit-image
                image = process_function(image, (image_shape[0], image_shape[1]))
                #print(image)
                #cv2.imshow("image", image)
                #print(" {} {} ".format(image.shape, labels[i]))        
                #cv2.waitKey()        
                #create a feature                
                feature = {'image': TFRecord._bytes_feature(tf.compat.as_bytes(image.tostring())),
                        'label': TFRecord._int64_feature(labels[i])}
                
                #create an example protocol buffer
                example = tf.train.Example(features = tf.train.Features(feature=feature))        
                #serialize to string and write on the file
                writer.write(example.SerializeToString())
                mean_image = mean_image + image / len(filenames)
            except ValueError :
                n_reading_error = n_reading_error + 1 
                print("Error reading {}:{}".format(n_reading_error, filenames[i]))
                                
        writer.close()
        sys.stdout.flush()
        return mean_image

    @staticmethod
    def process_batch_threads(thr_index, ranges, data_dir, filenames, labels, image_shape, tfr_filename, process_function = imgproc.resize_image):    
        #create tf-records    
        tfr_filename_batch = '{}_{}.tfrecords'.format(tfr_filename, thr_index)
        mean_filename_batch = '{}_{}_mean.npy'.format(tfr_filename, thr_index)    
        writer = tf.io.TFRecordWriter(tfr_filename_batch)
        #filenames and lables should  have the same size    
        assert len(filenames) == len(labels)
        mean_batch = np.zeros(image_shape, dtype=np.float32)
        n_reading_error = 0    
        batch_size = ranges[thr_index][1] - ranges[thr_index][0]
        count = 0
        for idx in np.arange(ranges[thr_index][0], ranges[thr_index][1]) :   
            try:
                image = Image.read_image(f"{data_dir}/{filenames[idx]}", image_shape[2]) #scikit-image
                image = process_function(image, (image_shape[0], image_shape[1]))                            
                feature = {
                    'image': TFRecord._bytes_feature(tf.compat.as_bytes(image.tostring())),
                    'label': TFRecord._int64_feature(labels[idx])}            
                #create an example protocol buffer
                example = tf.train.Example(features = tf.train.Features(feature=feature))        
                #serialize to string and write on the file
                writer.write(example.SerializeToString())
                mean_batch = mean_batch + image / batch_size;
                count = count + 1
                if count % 100 == 0:
                    print('{} Thread {} --> processing {} of {} [{}, {}]'.format(datetime.now(), thr_index, count, batch_size, ranges[thr_index][0], ranges[thr_index][1]))
            except ValueError :
                n_reading_error = n_reading_error + 1 
                print('Error reading {}:{}'.format(n_reading_error, filenames[idx]))
        
        writer.close()
        mean_batch.astype(np.float32).tofile(mean_filename_batch)
        print('Thread {} --> saving mean at {}'.format(mean_filename_batch, thr_index))
        sys.stdout.flush()

    @staticmethod
    def create_tfrecords_threads(data_dir, filenames, labels, image_shape, tfr_filename, process_function, n_threads):
        assert len(filenames) == len(labels) 
        #break whole dataset int batches according to the number of threads
        spacing = np.linspace(0, len(filenames), n_threads + 1).astype(np.int)
        ranges = []    
        for i in np.arange(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])
        
        #launch a thread for each batch
        print('Launching {} threads for spacings: {}'.format(n_threads, ranges))
        sys.stdout.flush()
        threads = []
        for thr_index in np.arange(len(ranges)):
            args = (thr_index, ranges, data_dir, filenames, labels, image_shape, tfr_filename, process_function)
            t = threading.Thread(target = TFRecord.process_batch_threads, args=args)
            t.start()
            threads.append(t)         
        #wait until all threads end        
        for idx, thread in enumerate(threads):        
            thread.join()
        print('*******All threads have finished!!*******')    
        #compute the mean image from those computed by each thread    
        for idx in range(n_threads) :
            mean_filename = '{}_{}_mean.npy'.format(tfr_filename, idx)
            if idx == 0 :        
                mean_image = np.reshape(np.fromfile(mean_filename,  dtype=np.float32), image_shape)
            else :
                mean_image = mean_image + np.reshape(np.fromfile(mean_filename,  dtype=np.float32), image_shape)
        mean_image = mean_image / n_threads
        return mean_image           

    @staticmethod
    def create(config, _type, processFun=imgproc.resize_image) :
        """ 
            data_dir: Folder where data is located (train.txt and test.txt should be found)
            type: 'train' | 'test' | 'all' (default='all')             
            im_shape: [H,W,C] of the input          
            processFun: processing function which depends on the problem we are dealing with
        """

        data_dir = config.dataset_dir
        image_shape = np.asarray((config.image_height, config.image_width, config.channels))
        n_threads = config.num_threads

        #------------- creating train data
        if (_type == 'train') or (_type == 'all') : 
            filenames, labels = File.read_data_from_file(data_dir, dataset = 'train', shuf = True)
            if config.use_multithreads:
                tfr_filename = os.path.join(data_dir, 'train')
                training_mean = TFRecord.create_tfrecords_threads(data_dir, filenames, labels, image_shape, tfr_filename, processFun, n_threads)
            else:        
                tfr_filename = os.path.join(data_dir, 'train.tfrecords')            
                training_mean = TFRecord.create_from_file(filenames, labels, image_shape, tfr_filename, processFun)
                
            print('train_record saved at {}.'.format(tfr_filename))
            #saving training mean
            mean_file = os.path.join(data_dir, "mean.dat")
            print("mean_file {}".format(training_mean.shape))
            training_mean.astype(np.float32).tofile(mean_file)
            print("mean_file saved at {}.".format(mean_file))
            #saving shape file    
            shape_file = os.path.join(data_dir, "shape.dat")
            image_shape.astype(np.int32).tofile(shape_file)
            print("shape_file saved at {}.".format(shape_file))  
        #-------------- creating test data    
        if (_type == 'test') or (_type == 'all') :
            filenames, labels = File.read_data_from_file(data_dir, dataset="test", shuf = True)
            if config.use_multithreads:
                tfr_filename = os.path.join(data_dir, 'test')
                TFRecord.create_tfrecords_threads(filenames, labels, image_shape, tfr_filename, processFun, n_threads)
            else :    
                tfr_filename = os.path.join(data_dir, "test.tfrecords")
                TFRecord.create_from_file(data_dir, filenames, labels, image_shape, tfr_filename, processFun)
            print("test_record saved at {}.".format(tfr_filename))    

    @staticmethod
    def parser_tfrecord(serialized_input, input_shape, mean_image, number_of_classes, with_augmentation = False):
        """ parser tf_record to be used for dataset mapping"""
        features = tf.io.parse_example([serialized_input],
                                features={
                                        'image': tf.io.FixedLenFeature([], tf.string),
                                        'label': tf.io.FixedLenFeature([], tf.int64)
                                        })
        #image
        #rgb_mean = [123.68, 116.779, 103.939]
        #rgb_std = [58.393, 57.12, 57.375]         
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, input_shape)
        #data augmentation
        #central crop
        #dataagumentation prob = 0.4
        if with_augmentation:
            data_augmentation_prob = 0.5
            prob = tf.random.uniform((), 0 ,1)
            if prob < data_augmentation_prob :
                #image = tf.image.flip_left_right(image)                
                #fraction = tf.random.uniform((), 0.5, 0.9, dtype = tf.float32)
                if prob < data_augmentation_prob * 0.5 :
                    image = tf.image.central_crop(image, central_fraction = 0.7)
                    image = tf.cast(tf.image.resize(image, (input_shape[0], input_shape[1])), tf.uint8)
                else :
                    image = tf.image.flip_left_right(image)                
                
            #TODO
        
        image = tf.cast(image, tf.float32)
        #image = (image - rgb_mean) / rgb_std
        image = image - mean_image
        
        #label
        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, depth = number_of_classes)
        label = tf.reshape(label, [number_of_classes])
        
        return image, label          
    
    
    @staticmethod
    def parser_tfrecord_siamese(serialized_input, input_shape, mean_image,  with_augmentation = False):
        """ parser tf_record to be used for dataset mapping """
        features = tf.io.parse_example([serialized_input],
                                features={
                                        'image': tf.io.FixedLenFeature([], tf.string),
                                        'label': tf.io.FixedLenFeature([], tf.int64)
                                        })
        #image
        #rgb_mean = [123.68, 116.779, 103.939]
        #rgb_std = [58.393, 57.12, 57.375]         
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, input_shape)
        #data augmentation
        #central crop
        #dataagumentation prob = 0.4
        if with_augmentation:
            data_augmentation_prob = 0.5
            prob = tf.random.uniform((), 0 ,1)
            if prob < data_augmentation_prob :
                #image = tf.image.flip_left_right(image)                
                #fraction = tf.random.uniform((), 0.5, 0.9, dtype = tf.float32)
                if prob < data_augmentation_prob * 0.5 :
                    image = tf.image.central_crop(image, central_fraction = 0.7)
                    image = tf.cast(tf.image.resize(image, (input_shape[0], input_shape[1])), tf.uint8)
                else :
                    image = tf.image.flip_left_right(image)                
                
            #TODO
        
        image = tf.cast(image, tf.float32)
        #image = (image - rgb_mean) / rgb_std
        image = image - mean_image
        
        #label
        label = tf.cast(features['label'], tf.int32)                
        return image, label