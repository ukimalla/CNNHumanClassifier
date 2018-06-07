import keras
import unittest
from keras.models import load_model
import numpy as np
from core.model_decoder import DataDecoder

if __name__ == "__main__":
    target = 370520
    model = load_model('./core/minisample3000_trained_cnn.h5')
    decoder = DataDecoder("D:/datalist/imdb_crop/", 3000, True)
    decoder.get_data_from_json("data.json")
    path = decoder.data[target]['path']
    name = decoder.data[target]['name']
    print("path : " + path + ", name : " + name)
    x_sample1, y_sample1 = decoder.decode_with_target("data.json", target)
    print("My prediction : ")
    print("[Age] : " + str(y_sample1[0]))
    print("[Gender] : " + str(y_sample1[1]))
    print("")

    x_in = np.array([x_sample1])
    y_out = model.predict(x_in)
    print("CNN prediction : ")
    print("[Age] : " + str(y_out[0][0]))
    print("[Gender] : " + str(y_out[0][1]))