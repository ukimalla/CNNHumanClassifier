import unittest
from ..model_decoder import ImageDecoder
import tensorflow as tf


class ImageDecoderTest(unittest.TestCase):
    def test_decoder(self):
        image_decoder = ImageDecoder("D:/datalist/imdb_crop/", width=300, height=200, channel_number=3)
        img = image_decoder.decode("93/nm0000093_rm3593506304_1963-12-18_1998.jpg")
        self.assertEqual(img.get_shape(), tf.TensorShape([200, 300, 3]))


if __name__ == "__main__":
    unittest.main()
