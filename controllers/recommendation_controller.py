from flask import Blueprint, request, jsonify, make_response, session
import pandas as pd
from recommender import recommend, recommend_season, get_clothes_info

recommendation_bp = Blueprint('recommendation', __name__)

@recommendation_bp.route("/info/<gender>/<clothes_name>",methods=['GET'])
def get_clothes_info_req(gender,clothes_name):
    if gender in ['man', 'woman']:
        info = get_clothes_info(gender,clothes_name)
        if info is None:
            return make_response(jsonify({"error":"unknown clothes name"}), 400)
        return make_response(info.to_json(orient='index',force_ascii=False), 200)
    else:
        return make_response(jsonify({"error":"unknown gender"}), 400)

@recommendation_bp.route('/recommend',methods=['POST'])
def recommend_response():
    required_fields = ['gender', 'age', 'color', 'faceshape', 'bodyshape', 'page']
    data = request.json
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return make_response(jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400)

    try:
        page = int(data['page'])
    except ValueError:
        return make_response(jsonify({"error": "Invalid page value"}), 400)

    is_final = False
    start = page * 50
    end = start + 50

    df = _get_recommendations_from_session('dataframe', recommend, data)

    if len(df) <= start:
        return make_response(jsonify({"error": "Page out of index"}), 404)

    if len(df) < end:
        end = len(df)
        is_final = True

    df_slice = df.iloc[start:end, :]
    json_data = df_slice.to_json(orient='index')
    response = make_response(json_data, 200)
    response.headers['isfinal'] = str(int(is_final))
    return response

@recommendation_bp.route('/recommend_season',methods=['POST'])
def recommend_season_response():
    required_fields = ['gender', 'age', 'color', 'faceshape', 'bodyshape', 'season', 'page']
    data = request.json
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return make_response(jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400)

    try:
        page = int(data['page'])
    except ValueError:
        return make_response(jsonify({"error": "Invalid page value"}), 400)

    is_final = False
    start = page * 50
    end = start + 50

    df = _get_recommendations_from_session('dataframe', recommend_season, data)

    if len(df) <= start:
        return make_response(jsonify({"error": "Page out of index"}), 404)

    if len(df) < end:
        end = len(df)
        is_final = True

    df_slice = df.iloc[start:end, :]
    json_data = df_slice.to_json(orient='index')
    response = make_response(json_data, 200)
    response.headers['isfinal'] = str(int(is_final))
    return response

def _get_recommendations_from_session(session_key, recommend_func, data):
    if session_key in session:
        df = pd.read_json(session[session_key], orient='index')
    else:
        df = recommend_func(
            data['gender'],
            data['age'],
            data['color'],
            data['faceshape'],
            data['bodyshape']
        ).reset_index(drop=True)
        if df is None:
            raise ValueError("Failed to generate recommendations")
        df.columns = ['image', 'predict', 'average', 'total']
        session[session_key] = df.to_json(orient='index')
    return df