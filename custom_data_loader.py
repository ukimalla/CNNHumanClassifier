import tensorflow as tf
import numpy as np
import codecs
import json
import threading
import matplotlib.image as mpimg

import time

import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import matplotlib.pyplot as plt

def load_imdb_data(height: int = 128, width: int = 128, channel: int = 3, n_inputs: int = 20000, startIndex: int = 0):

    f = codecs.open('data.json', 'r', 'utf-8-sig')
    data = json.load(f)
    total_data = len(data["data"])
    data = data["data"]

    f.close()

    if n_inputs < 0:
        n_inputs = total_data

    image_array = np.empty((height, width, channel))  # Initializing nparray to store the dataset
    img_counter = 1  # counter for number of images processed

    startTime = time.time()

    # Loop to iterate through JSON
    # for i in range(startIndex, startIndex + n_inputs):  # Iterating through JSON
    for i in range(0, 10):  # Iterating through JSON
        img_counter += 1  #Incrementing img_counter

        reader = tf.read_file(str("/Users/ukimalla/Downloads/imdb_crop/" + data[i]["path"]))
        image = tf.image.decode_jpeg(reader, 3, 1)
        image = tf.image.resize_images(image, [height, width])  # Resizing the image

        # Running tf session
        with tf.Session() as sess:
            tmp_array = sess.run(image)
            image_array = np.vstack([tmp_array, image_array])

        # Calculating the prop of images successfully imported to memory
        prop_complete = (img_counter / n_inputs) * 100
        if prop_complete % 2 == 0:
            print(str(prop_complete) + "% of dataset imported to memory")

        # If n_inputs have been uploaded to memory
        if img_counter > n_inputs or i >= total_data:
            break

    print(str(time.time() - startTime))



    image_array = image_array.reshape((int(image_array.shape[0] / height), height, width, 3))
    print(image_array.shape)

    # Display all the images

    for img_counter in range(0, image_array.shape[0] - 1):
        image = tf.Session().run(tf.cast(image_array[img_counter], tf.uint8))
        plt.imshow(image)
        plt.xlabel(str(data[img_counter]["age"]) + " " +  str(data[img_counter]["index"]) + str(data[img_counter]["path"]))
        plt.show()

    return image_array



def load_save_imdb_data(height: int = 128, width: int = 128, channel: int = 3, n_inputs: int = 20000):

    json_path = "data.json"
    imdb_path = "/home/ukimalla/Desktop/imdb_crop/"

    path, y_labels = load_filtered_data(json_path, min_score1=0)

    path = path[:n_inputs]
    y_labels = y_labels[:n_inputs]


    if n_inputs < 0:
        n_inputs = len(path)

    startTime = time.time()

    for i in range(0, len(path)):
        path[i] = str(imdb_path) + str(path[i])

    # TF Variables for importing and resizing images
    filename_queue = tf.train.string_input_producer(path)

    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)

    image = tf.image.decode_jpeg(image_file, channels=channel)
    image = tf.image.resize_images(image, [height, width])  # Resizing the image

    image.set_shape((height, width, 3))


    # Generate batch
    num_preprocess_threads = 4
    min_queue_examples = 10000


    images = tf.train.batch(
        [image, y_labels],
        batch_size=n_inputs,
        num_threads=num_preprocess_threads,
        # shapes=[(64, 64, 3), (n_inputs, 2)],
        capacity=50000)


    # Running tf session
    with tf.Session() as sess:

        # Required to get the filename matching to run.
        tf.initialize_all_variables().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_tensor, y_labels = sess.run(images)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)


    print(str(time.time() - startTime))

    # Display all the images
    for img_counter in range(0, len(image_tensor)):
        image = tf.Session().run(tf.cast(image_tensor[img_counter], tf.uint8))
        plt.imshow(image)
        plt.xlabel(str(y_labels[2][img_counter]))
        plt.show()


    return image_tensor, y_labels



def load_filtered_data(json_path: str = "data.json",  min_score1: float = 0):

    # Loading json to memory
    f = codecs.open(json_path, 'r', 'utf-8-sig')
    data = json.load(f)
    f.close()
    data = data["data"]
    mat_path = '/home/ukimalla/Desktop/imdb_crop/'

    pathList = []
    y_labels = []

    for d in data:
        score1 = d["score1"]
        score2 = d["score2"]
        age = d["age"]
        gender = d["gender"]
        path = d["path"]

        # Filtering through face score 1
        if str(score1).lower() == "nan" or str(score1).lower() == "-inf"\
                or str(score1).lower() == "inf" or score1 < min_score1:
            continue

        # Filter through face score 2 (If second score is present, skip)
        if str(score2).lower() != "nan":
            continue

        # Filtering through age
        if age > 100 or age < 0 \
                or str(age).lower() == "nan":
            continue

        # Filtering through gender
        if gender < 0 or gender > 1 \
                or str(gender).lower == "nan":
            continue

        pathList.append(path)
        y_labels.append([age, gender])

    y_labels = np.array(y_labels)

    return pathList, y_labels




def load_y_labels(n_labels: int = 10000):

    # JSON
    f = codecs.open('data.json', 'r', 'utf-8-sig')
    data = json.load(f)
    f.close()

    if(n_labels < 0):
        n_labels = len(data["data"])

    y_labels = []
    counter = 1
    for d in data["data"]:
        counter += 1
        y_labels.append([d["age"], d["gender"]])


    y_labels = np.array(y_labels)
    print(y_labels)

    outflie = open("data.npz", "wb")
    np.savez_compressed(outflie, y_labels)
    outflie.close()

    npzfile = np.load("data.npz")

    print(npzfile['arr_0'])
    print(len(data["data"]))

    return y_labels


def load_age_labels():

    # JSON
    f = codecs.open('data.json', 'r', 'utf-8-sig')
    data = json.load(f)
    f.close()

    y_labels = []
    counter = 1
    for d in data["data"]:
        counter += 1
        y_labels.append(d["age"])


    y_labels = np.array(y_labels)
    print(y_labels)

    outflie = open("age_labels.npz", "wb")
    np.savez_compressed(outflie, y_labels)
    outflie.close()

    npzfile = np.load("data.npz")

    print(npzfile['arr_0'])
    print(len(data["data"]))

    return y_labels


#
#
# load_y_labels(n_labels=-1)
#
# size_of_npz = 10

# for i in range(0, 5):
# image_array, label_array = load_save_imdb_data(n_inputs=size_of_npz, height=64, width=64)
    # np.savez_compressed("images part " + str(i) + ".npz", image_array)

#load_imdb_data(height=64, width=64, channel=3, n_inputs=10, startIndex=0)




image_tensor, y_labels = load_save_imdb_data(height=64, width=64, channel=3, n_inputs=5)

# image_array = load_save_imdb_data(height=64, width=64, channel=3, n_inputs=10)

#
# outflie = open("newdata.npz","wb")
# print("Writing to file")
# np.savez(outflie, image_array)
# outflie.close()
# print("Done Writing")
#
#
