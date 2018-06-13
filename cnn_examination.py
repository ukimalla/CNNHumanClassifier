import keras
import unittest
from keras.models import load_model
import numpy as np
from core.model_decoder import DataDecoder
import tensorflow as tf
from core.preprocessing import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    target = 370520
    model = load_model('train_categorical_final.h5')

    correct_age = 0
    correct_gender = 0
    n_samples = 0


    # Loading input data
    for filecount in tqdm(range(1, 8)):
        data = np.load("imdb_db_" + str(filecount) + "_of_7.npz")
        x_sample1 = data['x'].astype('float32')
        y_sample1 = data['y'].astype('float32')
        n_samples += len(x_sample1)

        x_sample1 /= 255

        bins = np.array([0, 10, 20, 25, 35, 45, 55, 65, 75, 100])

        y_sample1 = to_categorical(y_sample1, bins, exam=True)

        y_out = model.predict(x_sample1)



        for i in range(0, len(x_sample1)):
            if (np.allclose(y_sample1[0][i], y_out[0][i])):
                correct_age += 1
            if (y_sample1[1][i] == y_out[1][i][0]):
                correct_gender += 1

    print(correct_age/n_samples)
    print(correct_gender/n_samples)

    #
    #
    # for img_counter, image in enumerate(x_sample1):
    #     image_show = tf.Session().run(tf.cast(image, tf.uint8))
    #     plt.imshow(image_show)
    #     plt.xlabel(str(y_out[1][img_counter]))
    #     plt.show()
    #
    # #
    # # a = np.array([[12, 1]])
    # # b = np.array([[12, 1]])
    #
    # h = tf.Session().run(keras.metrics.binary_accuracy(y, y_out))
    # print(h)
    # h = tf.Session().run(keras.metrics.categorical_accuracy(y_sample1, y_out))
    # h = tf.Session().run(keras.metrics.sparse_categorical_accuracy(y_sample1, y_out))
    # print(h)