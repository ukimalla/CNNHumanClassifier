import tensorflow as tf
from .interface.decoder import Decoder
from tensorflow.python.framework.ops import Tensor
import codecs
import json


class ImageDecoder(Decoder):
    def __init__(self, path: str, channel_number: int=3, downscaling_ratio: int=1, width: int=192, height: int=192):
        self.channel = channel_number
        self.ratio = downscaling_ratio
        self.path = path
        self.width = width
        self.height = height

    def decode(self, img_path: str) -> Tensor:
        path = self.path + img_path
        reader = tf.read_file(path)
        image = tf.image.decode_jpeg(reader, self.channel, self.ratio)
        return tf.image.resize_images(image, [self.height, self.width])


class JsonDecoder:
    def __init__(self, json_file: str):
        f = codecs.open(json_file, 'r', 'utf-8-sig')
        data = json.load(f)
        f.close()
        self.size = data["count"]
        self.data = data["data"]



