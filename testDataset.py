import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta

import scipy.io as sio
import matplotlib.image as mpimg
import tables

import codecs
import json





file = sio.loadmat('/home/ukimalla/Downloads/imdb/imdb2.mat')
data = tables.open_file('/Users/ukimalla/Downloads/imdb/imdb2.mat')
table1 = data.root.imdb.value
varname = 'imdb'

dob = data.root.imdb.value.dob.value[:].astype('int32')
photo_taken = data.root.imdb.value.photo_taken.value[:].astype('int32')
data.root.imdb.value.full_path.value
d1 = int(dob[8][0]).value
d2 = int(photo_taken[8][0])
path =




file = tables.open_file('imdb.mat')

def get_age(dob, ptk):
    d = datetime.date(ptk, 6, 1)
    return int((d.toordinal() - dob)/365) + 1

matlab_datenum = d1

python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)



#
# for i in range(0, 10):
#
#     dob = int(data["dob"][0][0][0][i])
#     photo_taken = int(data["photo_taken"][0][0][0][i])
#
#     matlab_datenum = dob
#
#     python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
#
#     print(dob)
#     print(photo_taken)
#
#     gender = data["gender"][0][0][0][i]
#     age_INFO = get_age(dob, photo_taken)
#
#     name_INFO = data["name"][0][0][0][i][0]
#
#     if gender == 0:
#         gender = "Female"
#     else:
#         gender = "Male"
#
#     path_INFO = data["full_path"][0][0][0][i][0]
#
#     image = mpimg.imread("/Users/ukimalla/Downloads/imdb_crop/" + path_INFO)
#
#     info = str("ID:" + str(i) + " Name: " + name_INFO + " " + gender + " Age: " + str(age_INFO))
#
#
#     plt.imshow(image)
#     plt.xlabel(info)
#     plt.show()
#
#


f = codecs.open('data.json', 'r', 'utf-8-sig')
data = json.load(f)
f.close()
data = data["data"]


image = mpimg.imread("/Users/ukimalla/Downloads/imdb_crop/" + data[8]["path"])

info = str("ID:" + str(data[8]["index"]) + " Name: " + data[8]["name"] + " " + str(data[8]["gender"]) + " Age: " + str(data[8]["age"]))

plt.imshow(image)
plt.xlabel(info)
plt.show()
