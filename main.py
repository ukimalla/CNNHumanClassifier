import codecs
import json

f = codecs.open('data.json', 'r', 'utf-8-sig')
data = json.load(f)
f.close()


for d in data["data"]:
    print("index : " + str(d["index"]))
    print("path : " + str(d["path"]))
    print("name : " + str(d["name"]))
    print("gender : " + str(d["gender"]))
    print("age : " + str(d["age"]))
    print("face_location : " + str(d["face_location"]))
    print("\n")
