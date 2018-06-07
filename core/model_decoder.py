import tensorflow as tf
from .interface.decoder import Decoder
from .interface.decoder import MultiDecoder
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

    def decode(self, json_file: str) -> (ndarray, ndarray):
        f = codecs.open(json_file, 'r', 'utf-8-sig')
        data = json.load(f)
        f.close()
        json_data = data["data"]
        x_label = []
        y_label = []
        if self.random:
            random.shuffle(json_data)
        start = time.time()
        for i in range(0, self.limit):
            x_label.append(self.img_dec.decode(json_data[i]['path']))
            y_label.append([json_data[i]['age'], json_data[i]['gender']])
            if i % 100 == 0:
                print("progress : " + str(i/self.limit * 100) + "%")
        print("progress : 100%")
        end = time.time()
        print("elapsed time : " + str(end - start))
        return np.array(x_label), np.array(y_label)
