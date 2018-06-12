import tensorflow as tf
import numpy as np
import codecs
import json
from tqdm import tqdm
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

    # Display all the images with labels
    # Uncomment to enable
    # for img_counter in range(0, image_array.shape[0] - 1):
    #     image = tf.Session().run(tf.cast(image_array[img_counter], tf.uint8))
    #     plt.imshow(image)
    #     plt.xlabel(str(data[img_counter]["age"]) + " " +  str(data[img_counter]["index"])\
    #  + str(data[img_counter]["path"]))
    #     plt.show()
    #
    # return image_array



def load_save_imdb_data(height: int = 128, width: int = 128, channel: int = 3, start_index: int = 0, n_inputs: int = 20000):

    json_path = "data.json"
    imdb_path = "/home/ukimalla/Desktop/imdb_crop/"

    path, y_labels = load_filtered_data(json_path, min_score1=0)

    path = path[start_index:n_inputs + start_index]
    y_labels = y_labels[start_index:n_inputs + start_index]


    if n_inputs < 0:
        n_inputs = len(path)

    for i in range(0, len(path)):
        path[i] = str(imdb_path) + str(path[i])


    startTime = time.time()

    path_tf = tf.convert_to_tensor(path)
    y_labels_tf = tf.convert_to_tensor(y_labels)

    path, labels = tf.train.slice_input_producer([path, y_labels_tf], shuffle=False)

    # TF Variables for importing and resizing images
    # filename_queue = tf.train.string_input_producer([path])

    # image_reader = tf.WholeFileReader()
    # _, image_file = image_reader.read(filename_queue)

    image_buffer = tf.read_file(path)

    image_decode = tf.image.decode_jpeg(image_buffer, channels=channel)
    image = tf.image.resize_images(image_decode, [height, width])  # Resizing the image


    # Generate batch
    num_preprocess_threads = 4
    min_queue_examples = 10000


    image.set_shape((height, width, 3))

    images, labels = tf.train.batch(
        [image, labels],
        batch_size=n_inputs,
        num_threads=num_preprocess_threads,
        capacity=50000)


    # Running tf session
    with tf.Session() as sess:

        # Required to get the filename matching to run.
        tf.initialize_all_variables().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_tensor, y_labels= sess.run([images, labels])

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

    print(str(n_inputs) + "images processed in " + str(time.time() - startTime) + " seconds.")

    # Display all the images(Uncomment to enable)
    # for img_counter in range(0, len(image_tensor)):
    #     image = tf.Session().run(tf.cast(image_tensor[img_counter], tf.uint8))
    #     plt.imshow(image)
    #     plt.xlabel(str(y_labels[img_counter]))
    #     plt.show()


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


def create_npz(filename: str, n_parts: int = 4, n_samples: int = -1):
    if n_samples < 2:
        n_samples = 181483

    file_size = int(n_samples/n_parts)
    for i in tqdm(range(6, n_parts)):
        print("Processing part " + str(i + 1) + " of " + str(n_parts) + ".")
        print("Resizing images and storing in memory.")
        image_tensor, y_labels = load_save_imdb_data(height=64, width=64, channel=3, start_index=i * file_size,
                                                     n_inputs=file_size)
        saveFileName = filename + "_" + str(i + 1) + "_of_" + str(n_parts) + ".npz"
        saveFile = open(saveFileName, 'wb')
        print("Saving data to file " + saveFileName)
        np.savez_compressed(saveFile, x=image_tensor, y=y_labels)
        saveFile.close()
        print("Part " + str(i + 1) + " of " + str(n_parts) + " complete.")

