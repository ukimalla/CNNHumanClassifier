import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from core.preprocessing import to_categorical, y_age_gender_split


if __name__ == "__main__":
    learning_rate = 0.001
    batch_size = 128
    num_classes = 2
    epochs = 100

    # Input image dimensions
    img_rows, img_cols, channels = 64, 64, 3

    input_shape = (img_cols, img_rows, channels)


    # Model definition

    inputs = Input(shape=(img_rows, img_cols, 3))
    layer = Conv2D(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape)(inputs)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.25)(layer)
    layer = Conv2D(64, (3, 3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(64, (3, 3), activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Flatten()(layer)
    layer = Dense(1000, activation='relu')(layer)
    layer = Dense(1000, activation='relu')(layer)
    layer = Dropout(0.5)(layer)

    predictions1 = Dense(9, activation='softmax')(layer)
    predictions2 = Dense(1, activation='sigmoid')(layer)

    # predictions = concatenate([predictions1, predictions2])

    model = Model(inputs=inputs, outputs=[predictions1, predictions2])

    optimizer = keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss=["categorical_crossentropy", "binary_crossentropy"],
                  optimizer=optimizer,
                  metrics=['accuracy'])


    # DATA
    # Loading input data
    for i in range(1, 8):
        data = np.load("../imdb_db_" + str(1) + "_of_7.npz")
        X = data['x']
        y = data['y']
        X /= 255

        bins = np.array([10, 20, 25, 35, 45, 55, 65, 75, 100])

        y = to_categorical(y, bins)

        # Train test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_train_age, y_train_gender, y_test_age, y_test_gender = y_age_gender_split(y_train, y_test)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        model.fit(X_train, [y_train_age, y_train_gender],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_test, [y_test_age, y_test_gender]))
        score = model.evaluate(X_test, [y_test_age, y_test_gender], verbose=0)

        print('Validation Sample loss:', score[0], score[1])
        print('Validation Sample accuracy:', score[2], score[3])

        model.save("train_categorical_" + str(i) + ".h5")

    print('Validation Sample loss:', score[0])
    print('Validation Sample accuracy:', score[1])

    model.save('train_categorical_final.h5')
