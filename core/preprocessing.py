from sklearn.preprocessing import OneHotEncoder
import numpy as np

def to_categorical(y_labels, bins):

    y_age = y_labels[:, 0]
    y_gender = y_labels[:, 1].astype('float32')

    y_labels = np.digitize(y_age, bins=bins, right=False).reshape(-1,1)

    ohe = OneHotEncoder()
    y_age = ohe.fit_transform(y_labels).toarray()

    y_labels = []

    for i in range(len(y_age)):
        y_labels.append([y_age[0], y_gender[0]])

    return np.array(y_labels)

def y_age_gender_split(y_train, y_test):
    # Returnns train_age, train_gender, test_age, test_gender
    return y_train[:, 0], y_train[:, 1], y_test[:, 0], y_test[:, 1]