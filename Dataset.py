from TFRecord import TFRecord
import imgproc as imgproc

class Dataset:
    def __init__(self, config) -> None:
        self.config = config

    def create_tfrecords(self, type_dataset=None, image_type=None, process_fun=imgproc.process_image):
        if image_type == 'SKETCH': 
            process_fun = imgproc.process_sketch        
        elif image_type == 'MNIST':
            process_fun = imgproc.process_mnist

        TFRecord.create(self.config, type_dataset, processFun=process_fun)
