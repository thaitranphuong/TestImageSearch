from flask import Flask, request, jsonify
from flask_cors import CORS

import math
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import  Model

from PIL import Image
import pickle
import numpy as np

# Ham tao model
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
    return extract_model

# Ham tien xu ly, chuyen doi hinh anh thanh tensor
def image_preprocess(img):
    img = img.resize((224,224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("Xu ly : ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Trich dac trung
    vector = model.predict(img_tensor)[0]
    # Chuan hoa vector = chia chia L2 norm (tu google search)
    vector = vector / np.linalg.norm(vector)
    return vector

# Hàm thêm dữ liệu mới vào tập vectors và paths
def add_new_data(vectors, paths, model, image_path):
    new_vector = extract_vector(model, image_path)
    new_vector = new_vector.reshape(1, -1)
    vectors = np.concatenate((vectors, new_vector), axis=0)
    paths.append(image_path)
    return vectors, paths

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'application/json'

@app.route('/find-similar-images', methods=['POST'])
def find_similar_images():
    # Khoi tao model
    model = get_extract_model()

    # Load 4700 vector tu vectors.pkl ra bien
    vectors = pickle.load(open("vectors.pkl","rb"))
    paths = pickle.load(open("paths.pkl","rb"))

    # Nhận đường dẫn đến ảnh từ request
    search_image_path = request.get_data(as_text=True)

    # Trích xuất vector của ảnh đầu vào
    search_vector = extract_vector(model, search_image_path)

    # Tính khoảng cách từ search_vector đến tất cả các vector trong tập vectors
    distance = np.linalg.norm(vectors - search_vector, axis=1)

    # Sắp xếp và lấy ra K vector có khoảng cách ngắn nhất
    K = 2
    ids = np.argsort(distance)[:K]

    # Tạo danh sách đường dẫn và khoảng cách tương ứng
    result = [{'path': str(paths[id]), 'distance': str(distance[id])} for id in ids]

    # # Trả về kết quả dưới dạng JSON
    return jsonify(result)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    # Nhận đường dẫn từ body của request
    received_data = request.get_data(as_text=True)

    # Thực hiện đào tạo lại mô hình ở đây
    # Dinh nghia anh can tim kiem
    search_image = received_data

    # Khoi tao model
    model = get_extract_model()

    # Học tăng cường
    # Load 4700 vector tu vectors.pkl ra bien
    vectors = pickle.load(open("vectors.pkl", "rb"))
    paths = pickle.load(open("paths.pkl", "rb"))

    # Them du lieu moi vao
    vectors, paths = add_new_data(vectors, paths, model, search_image)

    # Lưu lại tập vectors và paths sau khi thêm dữ liệu mới
    pickle.dump(vectors, open("vectors.pkl", "wb"))
    pickle.dump(paths, open("paths.pkl", "wb"))

    # Trả về phản hồi
    return 'Model retrained successfully'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
