import tensorflow as tf
import numpy as np
import codecs
import json
import time


import matplotlib.pyplot as plt

def load_imdb_data(height: int = 128, width: int = 128, channel: int = 3, n_inputs: int = 20000):

    f = codecs.open('data.json', 'r', 'utf-8-sig')
    data = json.load(f)
    f.close()

    if n_inputs < 0:
        n_inputs = len(data["data"])
        print (len(data["data"]))

    image_array = np.empty((height, width, channel))  # Initializing nparray to store the dataset
    img_counter = 1  # counter for number of images processed

    startTime = time.time()

    # Loop to iterate through JSON
    for d in data["data"]:  # Iterating through JSON

        img_counter += 1  #Incrementing img_counter

       #print(img_counter)

        path = str("/Users/ukimalla/Downloads/imdb_crop/" + d["path"])
        reader = tf.read_file(path)
        image = tf.image.decode_jpeg(reader, 3, 1)
        image = tf.image.resize_images(image, [height, width])  # Resizing the image

        # Running tf session
        with tf.Session() as sess:
            tmp_array = sess.run(image)
            image_array = np.vstack([tmp_array, image_array])

        prop_complete = (img_counter / n_inputs) * 100 # Calculating the prop of images successfully imported to memory
        if prop_complete % 10 == 0:
            print(str(prop_complete) + "% of dataset imported to memory")

        if img_counter > n_inputs: # If n_inputs have been uploaded to memory
            break

    print(str(time.time() - startTime))

    image_array = image_array.reshape((int(image_array.shape[0] / height), height, width, 3))
    print(image_array.shape)

    # Display all the images
    '''
    for img_counter in range(0, image_array.shape[0] - 1):
        image = tf.Session().run(tf.cast(image_array[img_counter], tf.uint8))
        plt.imshow(image)
        plt.show()
    '''

    return image_array







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
        prop_complete = (counter / n_labels) * 100 # Calculating the prop of images successfully imported to memory
        if (prop_complete % 4 == 0):
            print("Prop complete: " + str(prop_complete))

    y_labels = np.array(y_labels)
    print(y_labels)

    outflie = open("data.npz", "wb")
    np.savez_compressed(outflie, y_labels)
    outflie.close()

    npzfile = np.load("data.npz")

    print(npzfile['arr_0'])













#load_y_labels(n_labels=-1)

image_array = load_imdb_data(n_inputs=100, height=64, width=64)
outflie = open("data.npz","wb")
np.savez_compressed(outflie, image_array)
outflie.close()

npzfile = np.load("data.npz")
npzfile['arr_0']
