import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D
import numpy as np


if __name__ == "__main__":
    learning_rate = 0.001
    batch_size = 128
    num_classes = 2
    epochs = 3


    # input image dimensions
    img_rows, img_cols, channels = 64, 64, 3

    input_shape = (img_cols, img_rows, channels)

    # the data, split between train and test sets
    data = np.load("./minisample3000.npz")
    x = data['x']
    y = data['y']

    x = np.split(x, [2400, 2900])
    y = np.split(y, [2400, 2900])

    x_train = x[0]
    y_train = y[0]
    x_test = x[1]
    y_test = y[1]
    x_validation = x[2]
    y_validation = y[2]

    print("x_train count : " + str(len(x_train)))
    print("y_train count : " + str(len(y_train)))
    print("x_test count : " + str(len(x_test)))
    print("y_test count : " + str(len(y_test)))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_validation = x_validation.astype('float32')

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    y_validation = y_validation.astype('float32')



    print("train count : " + str(x_train.shape[0]))
    print("test count : " + str(x_test.shape[0]))
    print("validation count : " + str(x_validation.shape[0]))



    inputs = Input(shape=(img_rows, img_cols, 3))
    x = Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape)(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions1 = Dense(1, activation='linear')(x)
    predictions2 = Dense(1, activation='sigmoid')(x)

    predictions = concatenate([predictions1, predictions2])

    model = Model(inputs=inputs, outputs=predictions)

    optimizer = keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_validation, y_validation, verbose=0)
    print('Validation Sample loss:', score[0])
    print('Validation Sample accuracy:', score[1])

    model.save('train1.h5')
