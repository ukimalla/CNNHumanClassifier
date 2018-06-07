import tensorflow as tf
from .interface.decoder import Decoder
from tensorflow.python.framework.ops import Tensor
import numpy as np
from numpy import ndarray
import codecs
import json
import time
import random


class ImageDecoder(Decoder):
    def __init__(self, path: str, channel_number: int=3, downscaling_ratio: int=1, width: int=64, height: int=64):
        self.channel = channel_number
        self.ratio = downscaling_ratio
        self.path = path
        self.width = width
        self.height = height
        self.sess = tf.Session()

    def decode(self, img_path: str) -> ndarray:
        path = self.path + img_path
        reader = tf.read_file(path)
        image = tf.image.decode_jpeg(reader, self.channel, self.ratio)
        return tf.image.resize_images(image, [self.height, self.width]).eval(session=self.sess)


class DataDecoder(Decoder):
    def __init__(self, img_path: str, limit=391197, random_sampling: bool=True):
        self.img_dec = ImageDecoder(img_path)
        self.limit = limit
        self.random = random_sampling
        self.data = None

    def decode(self, json_file: str) -> (ndarray, ndarray):
        self.get_data_from_json(json_file)
        x_label = []
        y_label = []
        if self.random:
            random.shuffle(self.data)
        start = time.time()
        for i in range(0, self.limit):
            x, y = self._decode(i)
            x_label.append(x)
            y_label.append(y)
            if i % 100 == 0:
                print("progress : " + str(i/self.limit * 100) + "%")
        print("progress : 100%")
        end = time.time()
        print("elapsed time : " + str(end - start))
        return np.array(x_label), np.array(y_label)

    def _decode(self, i: int) -> ('x', 'y'):
        return self.img_dec.decode(self.data[i]['path']), [self.data[i]['age'], self.data[i]['gender']]

    def decode_with_target(self, json_file: str, index: int) -> (ndarray, ndarray):
        self.get_data_from_json(json_file)
        return self._decode(index)

    def get_data_from_json(self, json_file:str):
        if self.data is None:
            f = codecs.open(json_file, 'r', 'utf-8-sig')
            data = json.load(f)
            f.close()
            self.data = data['data']
        else:
            pass


