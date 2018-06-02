import scipy.io as sio
import datetime
import json
import math


def get_age(dob, ptk):
    d = datetime.date(ptk, 6, 1)
    return int((d.toordinal() - dob)/365) + 1


def process_face_location(f: str) -> []:
    return str(f)[1:-1].split()


def data_filter(gender: str, score1: str, score2: str):
    try:
        if str(score1) != "-inf":
            return [int(gender), float(score1), float(score2)]
        else:
            return None
    except ValueError:
        return None


def main():
    file = sio.loadmat('../imdb.mat')
    data = file['imdb'][0][0]
    mydata = []
    for i in range(0, 460723):
        gender = data[3][0][i]
        score1 = data[6][0][i]
        score2 = data[7][0][i]
        filtered = data_filter(gender, score1, score2)
        if filtered is None:
            continue
        gender_INFO = filtered[0]
        score1_INFO = filtered[1]
        score2_INFO = filtered[2]

        dob = int(data[0][0][i])
        photo_taken = int(data[1][0][i])
        age_INFO = get_age(dob, photo_taken)
        path_INFO = data[2][0][i][0]
        name_INFO = data[4][0][i][0]
        face_location_INFO = process_face_location(data[5][0][i][0])

        mydata.append(
            {
                "index": i,
                "name": name_INFO,
                "path": path_INFO,
                "gender": gender_INFO,
                "age": age_INFO,
                "face_location": face_location_INFO,
                "score1": score1_INFO,
                "score2": score2_INFO
            }
        )
    count = len(mydata)
    json_data = {"count": count, "data": mydata}
    f = open("dump.json", "w")
    f.write(json.dumps(json_data, indent=4))
    f.close()


if __name__ == "__main__":
    main()






