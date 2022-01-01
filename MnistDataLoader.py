#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
#from keras.datasets import mnist
import numpy
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels


    def myFunc(self,a):
        return a.flatten()


    def load_data(self):
        f = lambda a : a.flatten()
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        x_train = (((numpy.array(x_train))/255) - 0.5)
        x_test = (((numpy.array(x_test))/255) - 0.5)


        # x_train = numpy.apply_along_axis(f,1,x_train)
        x_train = np.array([a.flatten() for a in x_train]).transpose()
        x_test = np.array([a.flatten() for a in x_test]).transpose()
        print(x_train.shape)
        return (x_train, y_train), (x_test, y_test)