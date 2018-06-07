import unittest
from ..model_decoder import DataDecoder
import tensorflow as tf
import numpy as np


class DataDecoderTest(unittest.TestCase):
    def test_decoder(self):
        data_decoder = DataDecoder("D:/datalist/imdb_crop/", 3000, True)
        x, y = data_decoder.decode("data.json")
        np.savez_compressed("./minisample3000.npz", x=x, y=y)
        npz_file = np.load("./minisample3000.npz")
        self.assertEqual(len(npz_file['x']),3000)


if __name__ == "__main__":
    unittest.main()
