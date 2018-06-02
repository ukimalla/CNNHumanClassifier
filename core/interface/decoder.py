import abc
from tensorflow.python.framework.ops import Tensor


class Decoder(abc.ABC):
    """
    The Role of Decoder:
       1. take a image from image_path
       2. output tensor which has fixed size.
    """
    @abc.abstractclassmethod
    def decode(self, img_path) -> Tensor:
        raise NotImplementedError()


class DataSet(abc.ABC):
    """
    DataSet should offer this method:
        1. read_data(image_path, json_file)
           Read data (img->(gender,age)) from json_file.
           DataSet must keep all of json_file as a tensor unit.
           but, It should not keep image_tensor, it will be read whenever "next_batch" is invoked.

        2. next_batch(batch_size) -> X_batch, Y_batch
           read n(=batch_size) data from dataSet,
           and return X_batch Tensor(shape=(height,width,color,n))
                      Y_batch Tensor(shape=(1,1,n))
    """
    @abc.abstractclassmethod
    def read_data(self, img_path, json_file):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def next_batch(self, batch_size) -> (Tensor, Tensor):
        raise NotImplementedError()

