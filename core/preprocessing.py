from sklearn.preprocessing import OneHotEncoder
import numpy as np

def to_categorical(y_labels, bins):

    y_age = y_labels[:, 0]
    y_gender = y_labels[:, 1]

    y_labels = np.digitize(y_age, bins=bins, right=False).reshape(-1,1)

    ohe = OneHotEncoder()
    y_labels = ohe.fit_transform(y_labels).toarray()

    # for i in range(len(y_labels)):
    #     y_labels[i].


    return y_labels, y_gender