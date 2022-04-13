import os

import miniaudio
from numpy import ndarray, array
from pymongo.collection import Collection
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from librosa import load, amplitude_to_db
from librosa.feature import melspectrogram
from flask import Flask, Response, send_from_directory, current_app
from flask import request
import numpy as np
from json import dumps
import geopy.distance
import io

model = load_model('c:/malure/model')
app = Flask(__name__)
class_names = ['anas_platyrhynchos', 'bombycilla_garrulus', 'corvus_cornix', 'corvus_frugilegus', 'crex_crex',
               'fringilla_coelebs', 'fulica_atra', 'glaucidium_passerinum', 'parus_major', 'pyrrhula_pyrrhula']
CONNECTION_STRING_PLACES = "mongodb://localhost:27017/"
CONNECTION_STRING_BIRDS = "mongodb://localhost:27017/"
from pymongo import MongoClient

places = MongoClient(CONNECTION_STRING_PLACES)

db = MongoClient(CONNECTION_STRING_BIRDS)


@app.route("/search", methods=['POST'])
def bird_search():
    rec = bytearray(request.data)
    result = dict()
    spec = amplitude_to_db(
        melspectrogram(y=array(rec, dtype=float), sr=48000, n_mels=30, n_fft=4800), ref=np.max)
    spectrogram_split = np.array_split(spec.T, np.arange(1000, spec.T.shape[0], 1000), axis=0)
    spectrogram_split[-1] = np.pad(spectrogram_split[-1], ((0, 1000 - spectrogram_split[-1].shape[0]), (0, 0)))

    for chunk in spectrogram_split:
        prediction = model.predict(chunk.reshape((1,) + chunk.shape), batch_size=128)
        for i, class_name in enumerate(class_names):
            print(prediction)
            if (prediction[i] > 0.1) and \
                    (class_name in result):
                result = result.update({class_name: float(max(prediction[i], result[class_name]))})
            elif prediction[i] > 0.1:
                result.update({class_name: float(prediction[i])})
    return Response(dumps(result),
                    mimetype='application/json')


@app.route("/get_place/<place_request>", methods=['GET'])
def place(place_request):
    try:
        resp = db.malure.places.find_one({"_id": place_request})
        if not (isinstance(resp, dict)):
            resp = ''
    except:
        resp = ''
    return resp


@app.route("/get_bird/<get_bird>", methods=['GET'])
def bird(get_bird):
    try:
        resp = db.malure.birds.find_one({"_id": get_bird})
        if not (isinstance(resp, dict)):
            resp = ''
    # print(resp.find_one({"_id": "a"}))
    # resp.get_collection('malure')
    # print(type(birda))
    except:
        resp = ''
    return resp


@app.route("/get_place_near_me/<place>", methods=['GET'])
def place_near_me(place):
    try:
        lat = float(place.split(',')[0])
        lon = float(place.split(',')[1])
        resp = list(db.malure.places.find({'$and': [{'lat': {'$gt': lat - 2}}, {'lat': {'$lt': lat + 2}},
                                                {'lon': {'$gt': lon * 0.9}}, {'lon': {'$lt': lon * 1.11}}]}))
        for i, res in enumerate(resp):
            resp[i] |= {'dist':     geopy.distance.distance((lat, lon), (res['lat'], res['lon'])).km}

        resp = sorted(resp, key=lambda dist: geopy.distance.distance((lat, lon), (
            dist['lat'], dist['lon'])))  # Assume places.find works and returns a dict
    except:
        resp = ''

    return str(resp[:len(resp) // 2])


@app.route("/get_bird_pic/<bird>", methods=['GET'])
def bird_pic(bird):
    print(os.path.join(current_app.root_path, app.config['BIRD_PIC_FOLDER']))
    return send_from_directory(directory=os.path.join(current_app.root_path, app.config['BIRD_PIC_FOLDER']),
                               path=bird+".png")


@app.route("/get_place_pic/<place>", methods=['GET'])
def place_pic(place):
    return send_from_directory(directory=os.path.join(current_app.root_path, app.config['PLACE_PIC_FOLDER']),
                               path=place+".png")


app.config['JSON_AS_ASCII'] = False
app.config['BIRD_PIC_FOLDER'] = "c:/users/admin/birds/"
app.config['PLACE_PIC_FOLDER'] = "c:/users/admin/places/"
app.run(host='localhost', port=5000)
