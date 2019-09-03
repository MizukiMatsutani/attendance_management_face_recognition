# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition
import numpy as np
from flask import Flask, jsonify, request, redirect
import base64
from PIL import Image
from io import BytesIO
import cv2
import random, string
import os
import json

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_image(enc_image):
    dec_data = base64.b64decode(enc_image)
    dec_img= np.frombuffer(dec_data,dtype=np.uint8)
    
    return cv2.imdecode(dec_img, cv2.IMREAD_COLOR)

def randomname(n):
   return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

@app.route('/', methods=['GET'])
def index():
    fin=open('index.html')
    line=fin.read()
    fin.close()
    return line

@app.route('/api/upload', methods=['POST'])
def api_upload():
    enc_data  = request.get_json()['file']
    img = decode_image(enc_data)
    # dec_img = Image.open(BytesIO(dec_data))

    cv2.imwrite(r"images/target.png",img)

    if enc_data == "":
        return jsonify({"error": 422})

    name = detect_faces_in_image(r"images/target.png")

    return jsonify({"name": name})

##
# 解析
##
def detect_faces_in_image(file_stream):
    # 学習データ取得
    known_faces = get_known_face()
    known_face_encodings = known_faces[0]
    known_face_names = known_faces[1]

    # アップロードされたデータを変換
    img = face_recognition.load_image_file(file_stream)
    unknown_face_encodings = face_recognition.face_encodings(img)

    name = "Unknown"

    if len(unknown_face_encodings) > 0:
        face_encoding = unknown_face_encodings[0]
        ## TODO: なぜかJSONの一人目しか判定してくれない・・・
        match_results = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if match_results[0]:
            # return known_face_names[1]
            # 名前取得
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if match_results[best_match_index]:
                name = known_face_names[best_match_index]

    return name

##
# 判定用のエンコード済みデータ取得
##
def get_known_face():

    json_path = "encoded_images.json"
    path = "images/"
    with open(json_path, encoding='unicode-escape') as f:
	    jsons = json.loads(f.read())

    known_face_encodings = []
    known_face_names = []
    
    for name, encode in jsons:
        known_face_encodings.append(encode)
        known_face_names.append(name)

    return [known_face_encodings, known_face_names]

@app.route('/api/store', methods=['POST'])
def api_store():
    user_name = request.get_json()['user_name']
    enc_data  = request.get_json()['file']
    img = decode_image(enc_data)

    file_name = randomname(10)
    dirname = 'images'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    full_file_name = os.path.join(dirname, file_name + ".png")
    cv2.imwrite(full_file_name, img)

    remake_face_encode_json(full_file_name, user_name)

    return jsonify({"user_name": user_name})

## 判定用のjson再作成
def remake_face_encode_json(file_name, user_name):
    json_path = "encoded_images.json"

    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="UTF-8") as f:
            pass

    with open(json_path, encoding='unicode-escape') as f:
        jsons = json.loads(f.read())

    encode = []
    
    for row in jsons:
        encode.append(row)

    enc_image = face_recognition.load_image_file(file_name)
    face_encoding = face_recognition.face_encodings(enc_image)[0]
    encode.append([user_name, face_encoding.tolist()])

    # jsons.append([user_name, "aaaaaaa"])
    
    with open(json_path, 'w', encoding="UTF-8") as fw:
        json.dump(encode, fw, indent=4)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)