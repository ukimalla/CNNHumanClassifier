import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    #constant
    test_ratio = 0.2
    validation_ratio = 0.05

    #parameter
    batch_size = 128
    num_classes = 2
    epochs = 3
    learning_rate = 0.001



    # input image dimensions
    img_rows, img_cols, channels = 64, 64, 3

    input_shape = (img_cols, img_rows, channels)

    # the data, split between train and test sets
    data = np.load("D:/sample/data80000_part3.npz")
    x = data['x']
    y = data['y']

    _test_dataindex = int((1-test_ratio)*len(x))
    _validation_dataindex = int((1-validation_ratio)*len(x))

    # x = np.split(x, [_test_dataindex, _validation_dataindex])
    # y = np.split(y, [_test_dataindex, _validation_dataindex])

    x_train = x[:_test_dataindex]
    y_train = y[:_test_dataindex]
    x_test = x[_test_dataindex:_validation_dataindex]
    y_test = y[_test_dataindex:_validation_dataindex]
    x_validation = x[_validation_dataindex:]
    y_validation = y[_validation_dataindex:]

    x_train /= 255
    x_test /= 255
    x_validation /= 255

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_validation = x_validation.astype('float32')

    # y_train = y_train.astype('float32')
    # y_test = y_test.astype('float32')
    # y_validation = y_validation.astype('float32')

    print("train count : " + str(x_train.shape[0]))
    print("test count : " + str(x_test.shape[0]))
    print("validation count : " + str(x_validation.shape[0]))

    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    '''

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

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    print("model has been compiled")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_validation, y_validation, verbose=0)

    print('Validation Sample loss:', score[0])
    print('Validation Accuracy:', score[1])

    model.save('sample80000_trained_cnn.h5')