import abc
from tensorflow.python.framework.ops import Tensor
import numpy as np
from numpy import ndarray


class Decoder(abc.ABC):
    """
    The Role of Decoder:
       1. take a image from image_path
       2. output tensor which has fixed size.
    """
    @abc.abstractclassmethod
    def decode(self, file_path) -> ndarray:
        raise NotImplementedError()


