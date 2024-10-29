import pandas as pd
import numpy as np
import pickle
import joblib
import traceback
from sklearn.ensemble import RandomForestRegressor

'''
- 퍼스널컬러: spring,summer,autumn,winter
- 얼굴형: 'Heart, Oblong, Oval, Round, Square'
- (남자)체형 : TRAPEZOID,ROUND,RECTANGLE,INVERTED_TRIANGLE,TRIANGLE
- (여자)체형 : HOURGLASS,ROUND,RECTANGLE,INVERTED_TRIANGLE,TRIANGLE
'''

def get_clothes_info(gender, clothes_name):
    if gender == 'man':
        clothes_path = 'models/preprocessed_data/TL_man_clothes_2019.csv'
    elif gender == 'woman':
        clothes_path = 'models/preprocessed_data/TL_woman_clothes_2019.csv'
    else:
        return None

    df_clothes = pd.read_csv(clothes_path)
    df_clothes = df_clothes[df_clothes['image'] == clothes_name]
    if df_clothes.empty:
        return None
    return df_clothes.iloc[0:1, :]

def recommend(gender, age, color, face, body):
    try:
        age_group = _get_age_group(age)
        if gender.lower() == 'man':
            return _recommend_internal(
                'models/preprocessed_data/TL_man_clothes_2019.csv',
                'models/preprocessed_data/TL_man_rating_2019.csv',
                'models/model_files/onehot_encoder_man.pkl',
                'models/model_files/random_man.pkl',
                '남성',
                age_group,
                color.lower(),
                face.lower(),
                body.lower()
            )
        elif gender.lower() == 'woman':
            return _recommend_internal(
                'models/preprocessed_data/TL_woman_clothes_2019.csv',
                'models/preprocessed_data/TL_woman_rating_2019.csv',
                'models/model_files/onehot_encoder_woman.pkl',
                'models/model_files/random_woman.pkl',
                '여성',
                age_group,
                color.lower(),
                face.lower(),
                body.lower()
            )
        else:
            return None
    except Exception as e:
        print(traceback.format_exc())
        return None

# season: summer,winter,spring
def recommend_season(gender, age, color, face, body, season):
    season_map = {'summer': '여름', 'winter': '겨울', 'spring': '봄/가을'}
    season_korean = season_map.get(season.lower(), '봄/가을')
    try:
        age_group = _get_age_group(age)
        if gender.lower() == 'man':
            return _recommend_internal(
                'models/preprocessed_data/TL_man_clothes_2019.csv',
                'models/preprocessed_data/TL_man_rating_2019.csv',
                'models/model_files/onehot_encoder_man.pkl',
                'models/model_files/random_man.pkl',
                '남성',
                age_group,
                color.lower(),
                face.lower(),
                body.lower(),
                season_korean
            )
        elif gender.lower() == 'woman':
            return _recommend_internal(
                'models/preprocessed_data/TL_woman_clothes_2019.csv',
                'models/preprocessed_data/TL_woman_rating_2019.csv',
                'models/model_files/onehot_encoder_woman.pkl',
                'models/model_files/random_woman.pkl',
                '여성',
                age_group,
                color.lower(),
                face.lower(),
                body.lower(),
                season_korean
            )
        else:
            return None
    except Exception as e:
        print(traceback.format_exc())
        return None

def _get_age_group(age):
    age = int(age) // 10 * 10
    if age < 20:
        age = 20
    elif age >= 60:
        age = 50
    return f"{age}대"

def _recommend_internal(clothes_path, rating_path, encoder_path, model_path, gender, age, color, face, body, season=None):
    reg = joblib.load(model_path)
    df_user = pd.DataFrame({
        'r_gender': [gender],
        'age': [age],
        'personal_color': [color],
        'faceshape': [face],
        'bodyshape': [body]
    })
    df_clothes = pd.read_csv(clothes_path)
    if season:
        df_clothes = df_clothes[df_clothes['어울리는계절'] == season].reset_index(drop=True)
    df_rating = pd.read_csv(rating_path).groupby('image').mean()[['선호여부']]
    df_rating.columns = ['평균선호']
    df = pd.concat([df_user, df_clothes], axis=1).ffill()
    df_clothes_name = df['image']
    df = df.drop(columns=['image'])

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    df_encoded = encoder.transform(df.loc[:, 'r_gender':'분위기'])
    df_encoded = pd.DataFrame(
        df_encoded,
        columns=[f"col{i}_{elem}" for i, sublist in enumerate(encoder.categories_) for elem in sublist]
    )
    df_test = pd.concat([df_encoded, df.loc[:, '멋있다':].astype(np.int8)], axis=1)

    predict = reg.predict(df_test)
    df_recommend = pd.DataFrame({
        'image': df_clothes_name,
        '예상평점': predict
    }).sort_values(by='예상평점', ascending=False).head(100)
    df_recommend = pd.merge(df_recommend, df_rating, how='inner', on='image')
    df_recommend['종합평점'] = df_recommend['예상평점'] * 0.7 + df_recommend['평균선호'] * 0.3
    df_recommend = df_recommend.sort_values(by='종합평점', ascending=False)
    return df_recommend

def update_model(gender, age, color, faceshape, bodyshape, clothes, rating):
    # 성별 처리
    if gender.lower() == 'man':
        gender_kor = '남성'
        clothes_path = 'models/preprocessed_data/TL_man_clothes_2019.csv'
        model_path = 'models/model_files/random_man.pkl'
        train_x_path = 'train/train_x_man.csv'
        train_y_path = 'train/train_y_man.csv'
        encoder_path = 'models/model_files/onehot_encoder_man.pkl'
        rating_path = 'models/preprocessed_data/TL_man_rating_2019.csv'
    elif gender.lower() == 'woman':
        gender_kor = '여성'
        clothes_path = 'models/preprocessed_data/TL_woman_clothes_2019.csv'
        model_path = 'models/model_files/random_woman.pkl'
        train_x_path = 'train/train_x_woman.csv'
        train_y_path = 'train/train_y_woman.csv'
        encoder_path = 'models/model_files/onehot_encoder_woman.pkl'
        rating_path = 'models/preprocessed_data/TL_woman_rating_2019.csv'
    else:
        print('Invalid gender')
        return

    try:
        # 연령대 그룹화
        age_group = _get_age_group(age)

        ratings = pd.read_csv(rating_path)
        ratings = pd.concat([ratings, pd.DataFrame({'R_id': [0], 'image': [clothes], '선호여부': [rating], '스타일선호': [True]})], ignore_index=True)
        ratings.to_csv(rating_path, index=False)

        train_x = pd.read_csv(train_x_path)
        train_y = pd.read_csv(train_y_path)

        df_user = pd.DataFrame({
            'r_gender': [gender_kor],
            'age': [age_group],
            'personal_color': [color.lower()],
            'faceshape': [faceshape.lower()],
            'bodyshape': [bodyshape.lower()]
        })
        df_clothes = pd.read_csv(clothes_path)
        df_clothes = df_clothes[df_clothes['image'] == clothes].reset_index(drop=True)
        df = pd.concat([df_user, df_clothes], axis=1)
        df = df.drop(columns=['image'])

        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        df_encoded = encoder.transform(df.loc[:, 'r_gender':'분위기'])
        df_encoded = pd.DataFrame(df_encoded, columns=[f"col{i}_{elem}" for i, sublist in enumerate(encoder.categories_) for elem in sublist])
        df_append = pd.concat([df_encoded, df.loc[:, '멋있다':].astype(np.int8)], axis=1)

        train_x = pd.concat([train_x, df_append], ignore_index=True)
        train_y = pd.concat([train_y, pd.DataFrame({'선호여부': [rating]})], ignore_index=True)['선호여부']

        train_x.to_csv(train_x_path, index=False)
        train_y.to_csv(train_y_path, index=False)

        reg = RandomForestRegressor(random_state=0, n_jobs=-1)
        reg.fit(train_x, train_y)
        joblib.dump(reg, model_path)
    except Exception as e:
        print('모델 업데이트에 실패했습니다:', e)
