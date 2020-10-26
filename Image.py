import os

import numpy as np
import skimage.io as io
import skimage.color as color
import skimage.transform as transf

import imgproc as imgproc

class Image:
    """ Helpers Images """

    @staticmethod
    def read_image(filename, number_of_channels):
        """ read_image using skimage
            The output is a 3-dim image [H, W, C]
        """
        if number_of_channels == 1:            
            image = io.imread(filename, as_gray = True)
            image = imgproc.toUINT8(image)
            assert(len(image.shape) == 2)
            image = np.expand_dims(image, axis = 2) #H,W,C                    
            assert(len(image.shape) == 3 and image.shape[2] == 1)
        elif number_of_channels == 3:
            image = io.imread(filename)
            if(len(image.shape) == 2):
                image = color.gray2rgb(image)
            elif image.shape[2] == 4:
                image = color.rgba2rgb(image) 
            image = imgproc.toUINT8(image)
            assert(len(image.shape) == 3 and image.shape[2] == 3)
        else:
            raise ValueError("number_of_channels must be 1 or 3")
        if not os.path.exists(filename):
            raise ValueError(filename + " does not exist!")
        return image

    @staticmethod
    def resize_image(image, imsize):
        """Resize an image.

        Parameters
        ----------
        image : str
            The image
        imsize : tuple
            The shape image (height, width)
        """
        image_out = transf.resize(image, imsize)    
        image_out = Image.toUINT8(image_out)
        return image_out

    @staticmethod
    def toUINT8(image):
        if image.dtype == np.float64:
            image = image * 255
        elif image.dtype == np.uint16:
            image = image >> 8

        image[image<0] = 0
        image[image>255] = 255

        image = image.astype(np.uint8, copy=False)

        return image