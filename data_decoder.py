import unittest
from core.model_decoder import DataDecoder
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

size_of_npz = 80000

tf.logging.set_verbosity(tf.logging.ERROR)

for i in range(0, 4):
    data_decoder = DataDecoder("I:/datalist/imdb_crop/", size_of_npz, True, seed=80811, start_index= size_of_npz * i)
    x, y = data_decoder.decode("data.json")
    del data_decoder
    np.savez_compressed("D:/sample/data" + str(size_of_npz)+ "_part" + str(i) + ".npz", x=x, y=y)
