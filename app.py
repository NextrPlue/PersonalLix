from flask import Flask, request,jsonify, send_file, make_response,session
import numpy as np
import cv2
import pandas as pd
import os
from datetime import timedelta
import redis
from rq import Queue

from get_face_shape import get_face_shape
from get_body_shape import get_body_shape
from recommender import recommend, recommend_season, get_clothes_info
from worker import update_data

'''
class BodyShape(Enum):
    HOURGLASS=0
    TRAPEZOID=1
    ROUND=2
    RECTANGLE=3
    INVERTED_TRIANGLE=4
    TRIANGLE=5

age: 20,30,40,50,60
gender: man, woman

faceshape = ['heart','oblong','oval','round','square']
color = ['spring', 'summer', 'autumn', 'winter']
'''


app = Flask(__name__)
app.secret_key = 'asdf92($(*()))8u983ij9s8eduf98s'
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=10)

redis_conn = redis.Redis()
task_queue = Queue(connection=redis_conn)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/info/<gender>/<clothes_name>",methods=['GET'])
def get_clothes_info_req(gender,clothes_name):
    if gender in ['man', 'woman']:
        info = get_clothes_info(gender,clothes_name)
        if info is None:
            return make_response(jsonify({"error":"unknown clothes name"}), 400)
        return make_response(info.to_json(orient='index',force_ascii=False), 200)
    else:
        return make_response(jsonify({"error":"unknown gender"}), 400)

@app.route('/upload',methods=['POST'])
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

@app.route('/recommend',methods=['POST'])
def recommend_response():
    if 'gender' not in request.json:
        return make_response(jsonify({"error": "No gender found"}), 400)
    if 'age' not in request.json:
        return make_response(jsonify({"error": "No age found"}), 400)
    if 'color' not in request.json:
        return make_response(jsonify({"error": "No color found"}), 400)
    if 'faceshape' not in request.json:
        return make_response(jsonify({"error": "No faceshape found"}), 400)
    if 'bodyshape' not in request.json:
        return make_response(jsonify({"error": "No bodyshape found"}), 400)
    if 'page' not in request.json: # page start at 0
        return make_response(jsonify({"error": "No page found"}), 400)
    gender = request.json['gender']
    age = request.json['age']
    color = request.json['color']
    faceshape = request.json['faceshape']
    bodyshape = request.json['bodyshape']
    page = int(request.json['page'])

    isFinal=0

    start = page*50
    end = start+50

    df=''
    if 'dataframe' in session:
        data = session['dataframe']
        df=pd.read_json(data,orient='index')
    else:
        df = recommend(gender,age,color,faceshape,bodyshape).reset_index(drop=True)
        if df is None:
            return make_response(jsonify({"error": "fail to recommend..."}), 500)
        df.columns = ['image','predict','average','total']
        session['dataframe'] = df.to_json(orient='index')



    if len(df)<=start:
        return make_response(jsonify({"error": f"page out of index: len(df) is {len(df)}"}), 404)

    if len(df)<end:
        end=len(df)
        isFinal=1
    # df: image, 예상평점, 평균선호, 종합평점
    df = df.iloc[start:end,:]

    json_data = df.to_json(orient='index')

    res =  make_response(json_data,200)
    res.headers['isfinal'] = str(isFinal)
    return res


@app.route('/recommend_season',methods=['POST'])
def recommend_season_response():
    if 'gender' not in request.json:
        return make_response(jsonify({"error": "No gender found"}), 400)
    if 'age' not in request.json:
        return make_response(jsonify({"error": "No age found"}), 400)
    if 'color' not in request.json:
        return make_response(jsonify({"error": "No color found"}), 400)
    if 'faceshape' not in request.json:
        return make_response(jsonify({"error": "No faceshape found"}), 400)
    if 'bodyshape' not in request.json:
        return make_response(jsonify({"error": "No bodyshape found"}), 400)
    if 'season' not in request.json:
        return make_response(jsonify({"error": "No season found"}), 400)
    if 'page' not in request.json: # page start at 0
        return make_response(jsonify({"error": "No page found"}), 400)
    gender = request.json['gender']
    age = request.json['age']
    color = request.json['color']
    faceshape = request.json['faceshape']
    bodyshape = request.json['bodyshape']
    season = request.json['season']
    page = int(request.json['page'])

    isFinal=0

    start = page*50
    end = start+50

    df=''
    if 'dataframe' in session:
        data = session['dataframe']
        df=pd.read_json(data,orient='index')
    else:
        df = recommend_season(gender,age,color,faceshape,bodyshape,season).reset_index(drop=True)
        if df is None:
            return make_response(jsonify({"error": "fail to recommend..."}), 500)
        session['dataframe'] = df.to_json(orient='index')



    if len(df)<=start:
        return make_response(jsonify({"error": f"page out of index: len(df) is {len(df)}"}), 404)

    if len(df)<end:
        end=len(df)
        isFinal=1
    # df: image, 예상평점, 평균선호, 종합평점
    df = df.iloc[start:end,:]
    json_data = df.to_json(orient='index')


    res = make_response(json_data,200)
    res.headers['isfinal'] = str(isFinal)
    return res

@app.route('/photo/<gender>/<image_name>', methods=['GET'])
def get_photo(gender, image_name):
    # category와 image_name을 기반으로 이미지 경로 설정
    image_dir=''
    if gender=='man':
        image_dir = '../FashionWebp/man'
    else:
        image_dir = '../FashionWebp/woman'

    if image_name.split('.')[-1] == 'jpg':
        image_name = image_name.split('.')[0] + '.' + 'webp'

    image_path = os.path.join(image_dir, image_name)

    # 이미지 파일이 존재하는지 확인
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/webp')
    else:
        # 이미지가 없으면 404 에러 반환
        return make_response(jsonify({"error": "image not found"}), 404)

@app.route("/clear")
def session_out():

    session.clear()
    return make_response(jsonify({"info": "session clear"}), 200)
@app.route('/feedback', methods=['POST'])
def updade_model_req():
    data = request.json

    if 'gender' not in data:
        return make_response(jsonify({"error": "No gender found"}), 400)
    if 'age' not in data:
        return make_response(jsonify({"error": "No age found"}), 400)
    if 'color' not in data:
        return make_response(jsonify({"error": "No color found"}), 400)
    if 'faceshape' not in data:
        return make_response(jsonify({"error": "No faceshape found"}), 400)
    if 'bodyshape' not in data:
        return make_response(jsonify({"error": "No bodyshape found"}), 400)
    if 'clothes' not in data:
        return make_response(jsonify({"error": "No clothes found"}), 400)
    if 'rating' not in data:
        return make_response(jsonify({"error": "No rating found"}), 400)

    job = task_queue.enqueue(update_data,data)
    return make_response(jsonify({"info": f"job submitted: {job.get_id()}"}), 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
