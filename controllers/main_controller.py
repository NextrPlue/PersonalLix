from flask import Blueprint, request, jsonify, make_response, session, send_file
import os
import numpy as np
import cv2
from get_face_shape import get_face_shape
from get_body_shape import get_body_shape

main_bp = Blueprint('main_bp', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@main_bp.route('/upload',methods=['POST'])
def upload_file():
    required_files = ['face', 'body', 'body_handsup']
    for file_key in required_files:
        if file_key not in request.files:
            return make_response(jsonify({"error": f"No {file_key} part"}), 400)
        if request.files[file_key].filename == '':
            return make_response(jsonify({"error": f"No {file_key} photo"}), 400)
        if not allowed_file(request.files[file_key].filename):
            return make_response(jsonify({"error": "File type not allowed"}), 400)

    for form_key in ['gender', 'age']:
        if form_key not in request.form:
            return make_response(jsonify({"error": f"No {form_key} value"}), 400)

    gender = request.form['gender']
    try:
        age = int(request.form['age'])
    except ValueError:
        return make_response(jsonify({"error": "Invalid age value"}), 400)

    try:
        face_img = _read_image(request.files['face'])
        body_img = _read_image(request.files['body'])
        body_handsup_img = _read_image(request.files['body_handsup'])
    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 400)

    face_shape = get_face_shape(face_img)
    if face_shape is None:
        return make_response(jsonify({"error": "No face detected"}), 400)

    body_shape = get_body_shape(body_img, body_handsup_img, gender, age)
    if body_shape is None:
        return make_response(jsonify({"error": "Failed to determine body shape"}), 400)

    result = {
        'gender': gender,
        'age': age,
        'color': 'spring',  # Assuming default value; adjust as needed
        'faceshape': face_shape,
        'bodyshape': body_shape
    }
    return make_response(jsonify(result), 200)

def _read_image(file_storage):
    file_bytes = file_storage.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image {file_storage.filename}")
    return img


@main_bp.route('/photo/<gender>/<image_name>', methods=['GET'])
def get_photo(gender, image_name):
    valid_genders = ['man', 'woman']
    if gender not in valid_genders:
        return make_response(jsonify({"error": "Invalid gender"}), 400)

    image_dir = f'../FashionWebp/{gender}'
    if image_name.endswith('.jpg'):
        image_name = image_name.rsplit('.', 1)[0] + '.webp'
    image_path = os.path.join(image_dir, image_name)

    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/webp')
    else:
        return make_response(jsonify({"error": "Image not found"}), 404)

@main_bp.route("/clear")
def session_out():
    session.clear()
    return make_response(jsonify({"info": "session clear"}), 200)