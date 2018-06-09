import numpy as np

test_ratio = 0.2
validation_ratio = 0.05

data = np.load("D:/sample/data80000_part0.npz")
x = data['x']
y = data['y']

_test_dataindex = int((1-test_ratio)*len(x))
_validation_dataindex = int((1-validation_ratio)*len(x))

x_train = x[:_test_dataindex]
y_train = y[:_test_dataindex]
x_test = x[_test_dataindex:_validation_dataindex]
y_test = y[_test_dataindex:_validation_dataindex]
x_validation = x[_validation_dataindex:]
y_validation = y[_validation_dataindex:]

print(str(len(x_train)))
print(str(len(x_test)))
print(str(len(x_validation)))