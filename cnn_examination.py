import keras
import unittest
from keras.models import load_model
import numpy as np
from core.model_decoder import DataDecoder
import tensorflow as tf

if __name__ == "__main__":
    target = 370520
    model = load_model('train1.h5')



    # Loading input data
    data = np.load("/Users/ukimalla/Downloads/data390861_part0.npz")
    x_sample1 = data['x'][:3000].astype('float32')
    y_sample1 = data['y'][:3000].astype('float32')

    y_out = model.predict(x_sample1)
    #
    # for i in range(0, 3000):
    #     print("My prediction : ")
    #     print("[Age] : " + str(y_sample1[i][0]))
    #     print("[Gender] : " + str(y_sample1[i][1]))
    #
    #     print("CNN prediction : ")
    #     print("[Age] : " + str(y_out[i][0]))
#     print("[Gender] : " + str(y_out[i][1]))

    a = np.array([[12, 1]])
    b = np.array([[12, 1]])

    h = tf.Session().run(keras.metrics.binary_accuracy(a, b))
    print(h)
    h = tf.Session().run(keras.metrics.categorical_accuracy(y_sample1, y_out))
    h = tf.Session().run(keras.metrics.sparse_categorical_accuracy(y_sample1, y_out))
    print(h)