from flask import Flask, request,jsonify, send_file, make_response,session
from getFaceShape_efficientnet_crop_tflite import getFaceShape
from getBodyShape import getBodyShape
from random_forest import recommend
from random_forest import recommend_season
from random_forest import get_clothes_info
import numpy as np
import cv2
import pandas as pd
import os
from datetime import timedelta
import redis
from rq import Queue
from worker import update_data  # 작업 처리 함수

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
r = redis.Redis()
q = Queue(connection=r)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/info/<gender>/<clothes_name>",methods=['GET'])
def get_clothes_info_req(gender,clothes_name):
    if gender == 'man' or gender=='woman':
        info = get_clothes_info(gender,clothes_name)
        if info is None:
            return make_response(jsonify({"error":"unknown clothes name"}),400)
        return make_response(info.to_json(orient='index',force_ascii=False),200)
    else:
        return make_response(jsonify({"error":"unknown gender"}),400)



@app.route('/upload',methods=['POST'])
def upload_file():
    if 'face' not in request.files:
        return make_response(jsonify({"error":"No face part"}),400)
    if 'body' not in request.files:
        return make_response(jsonify({"error":"No body part"}),400)
    if 'body_handsup' not in request.files:
        return make_response(jsonify({"error":"No body_handsup part"}),400)
    
    face = request.files['face']
    body = request.files['body']
    body_handsup = request.files['body_handsup']

    if 'gender' not in request.form:
        return make_response(jsonify({"error":"No gender value"}),400)
    if 'age' not in request.form:
        return make_response(jsonify({"error":"No age value"}),400)

    gender = request.form['gender']
    age = request.form['age']
    age = int(age)
    

    if face and face.filename == '':
        return make_response(jsonify({"error":"No face photo"}),400)
    if body and body.filename == '':
        return make_response(jsonify({"error":"No body photo"}),400)
    if body_handsup and body_handsup.filename == '':
        return make_response(jsonify({"error":"No body_handsup photo"}),400)

    if allowed_file(face.filename) and allowed_file(body.filename) and allowed_file(body_handsup.filename):
        face_bytes = face.read()
        facearr = np.frombuffer(face_bytes,np.uint8)
        face_img = cv2.imdecode(facearr,cv2.IMREAD_COLOR)

        body_bytes = body.read()
        bodyarr = np.frombuffer(body_bytes,np.uint8)
        body_img = cv2.imdecode(bodyarr,cv2.IMREAD_COLOR)

        body_handsup_bytes = body_handsup.read()
        body_handsuparr = np.frombuffer(body_handsup_bytes,np.uint8)
        body_handsup_img = cv2.imdecode(body_handsuparr,cv2.IMREAD_COLOR)

        if face_img is None:
             return make_response(jsonify({"error": "Unable to read face image"}), 400)
        if body_img is None:
             return make_response(jsonify({"error": "Unable to read body image"}), 400)
        if body_handsup_img is None:
             return make_response(jsonify({"error": "Unable to read body_handsup image"}), 400)
        
        faceshape = getFaceShape(face_img)
        if faceshape is None:
            return make_response(jsonify({"error": "No Face detected"}), 400)
        # gender: man, woman
        # age: 20,30,40,50,60
        bodyshape = getBodyShape(body_img,body_handsup_img,gender,age)

        if bodyshape is None:
            return make_response(jsonify({"error": "Fail to check bodyshape"}), 400)
        
        result = {
            'gender': gender,
            'age': age,
            'color': 'spring',
            'faceshape': faceshape,
            'bodyshape': bodyshape

        }
        return make_response(jsonify(result),200)
    return make_response( jsonify({"error": "the file types are not allowed"}), 400)

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
    
    job = q.enqueue(update_data,data)
    return make_response(jsonify({"info": f"job submitted: {job.get_id()}"}), 200) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

#https://info-nemo.com/coding-story/wsl-wsl2-%EC%84%9C%EB%B2%84-%EC%99%B8%EB%B6%80-%EC%A0%91%EC%86%8D-%EC%8B%9C-%EC%A0%91%EC%86%8D-%EC%95%88%EB%90%98%EB%8A%94-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0/