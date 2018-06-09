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
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.feed_input = tf.placeholder(tf.string)
        self.processor = tf.read_file(self.feed_input)
        self.processor = tf.image.decode_jpeg(self.processor, self.channel, self.ratio)
        self.processor = tf.image.resize_images(self.processor, [self.height, self.width])

    def decode(self, img_path: str) -> ndarray:
        path = self.path + img_path
        return self.sess.run(self.processor, feed_dict={self.feed_input: path})


class DataDecoder(Decoder):
    def __init__(self, img_path: str, limit=390861, random_sampling: bool=True, start_index: int = 0, seed: int = -1):
        self.limit = limit
        self.start = start_index
        if self.start+self.limit>390861:
            self.limit = 390861-self.start
        self.random = random_sampling
        self.seed = seed
        self.data = None
        self.img_dec = ImageDecoder(img_path)
        self.x_label = []
        self.y_label = []

    def decode(self, json_file: str) -> (ndarray, ndarray):
        self.get_data_from_json(json_file)
        self.clear_label()
        if self.random:
            if self.seed == -1:
                random.shuffle(self.data)
            else:
                random.Random(self.seed).shuffle(self.data)
        print("start from " + str(self.start) + " to " + str(self.start+self.limit))
        start = time.time()
        for i in range(self.start, self.start+self.limit):
            x, y = self._decode(i)
            self.x_label.append(x)
            self.y_label.append(y)
            if (i-self.start) % 100 == 0:
                print("progress : " + str((i-self.start)/self.limit * 100) + "%")
        print("progress : 100%")
        end = time.time()
        print("elapsed time : " + str(end - start))
        self.x_label, self.y_label = np.array(self.x_label), np.array(self.y_label)
        return self.x_label, self.y_label

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

    def clear_label(self):
        del self.x_label
        del self.y_label
        self.x_label = []
        self.y_label = []


