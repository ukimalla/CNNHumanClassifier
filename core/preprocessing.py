from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

def to_categorical(y_labels, bins, exam: bool = False):
    y_age = y_labels[:, 0]
    y_gender = y_labels[:, 1].astype('float32')

    plt.hist(y_gender)
    plt.show()

    y_age = np.digitize(y_age, bins=bins, right=False).reshape(-1, 1)

    ohe = OneHotEncoder()
    y_age = ohe.fit_transform(y_age).toarray()

    if exam:
        return y_age, y_gender

    y_labels = []

    for i in range(len(y_age)):
        y_labels.append([y_age[i], y_gender[i]])

    return np.array(y_labels)


def y_age_gender_split(y_train, y_test):

    y_train_a = y_train[:, 0].tolist()
    y_train_gender = y_train[:, 1].tolist()
    y_test_a = y_test[:, 0].tolist()
    y_test_gender = y_test[:, 1].tolist()

    age_list = [y_train_a, y_test_a]

    y_train_age = []
    y_test_age = []

    for i, aL in enumerate(age_list):
        temp_list = []
        for age in aL:
            temp_list.append(age.tolist())
        if i == 0:
            y_train_age = temp_list
        else:
            y_test_age = temp_list

    # Returns train_age, train_gender, test_age, test_gender
    return np.asarray(y_train_age), np.array(y_train_gender), np.asarray(y_test_age), np.array(y_test_gender)
