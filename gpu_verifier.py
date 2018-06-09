from tensorflow.python.client import device_lib

print("Check if 'gpu' is in here")
print(device_lib.list_local_devices())