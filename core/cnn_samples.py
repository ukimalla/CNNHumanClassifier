import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    learning_rate = 0.001
    batch_size = 128
    num_classes = 2
    epochs = 3


    # Input image dimensions
    img_rows, img_cols, channels = 64, 64, 3

    input_shape = (img_cols, img_rows, channels)

    # Loading input data
    data = np.load("/Users/ukimalla/Downloads/data390861_part0.npz")
    X = data['x']
    y = data['y']
    len(y)




    # Train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    print("x_train count : " + str(len(X_train)))
    print("y_train count : " + str(len(y_train)))
    print("x_test count : " + str(len(X_test)))
    print("y_test count : " + str(len(y_test)))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    y



    print("train count : " + str(X_train.shape[0]))
    print("test count : " + str(X_test.shape[0]))



    inputs = Input(shape=(img_rows, img_cols, 3))
    layer = Conv2D(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape)(inputs)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.25)(layer)
    layer = Conv2D(64, (3, 3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Flatten()(layer)
    layer = Dense(1000, activation='relu')(layer)
    layer = Dropout(0.5)(layer)

    predictions1 = Dense(1, activation='linear')(layer)
    predictions2 = Dense(1, activation='sigmoid')(layer)

    predictions = concatenate([predictions1, predictions2])

    model = Model(inputs=inputs, outputs=predictions)

    optimizer = keras.optimizers.Adam(lr=learning_rate)


    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)


    b
    print('Validation Sample loss:', score[0])
    print('Validation Sample accuracy:', score[1])

    model.save('train1.h5')
