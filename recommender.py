import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import joblib
import traceback
'''
- 퍼스널컬러: spring,summer,autumn,winter
- 얼굴형: 'Heart, Oblong, Oval, Round, Square'
- (남자)체형 : TRAPEZOID,ROUND,RECTANGLE,INVERTED_TRIANGLE,TRIANGLE
- (여자)체형 : HOURGLASS,ROUND,RECTANGLE,INVERTED_TRIANGLE,TRIANGLE
'''
'''
r_gender,age,personal_color,faceshape,bodyshape
'''

def get_clothes_info(gender, clothes_name):
    if gender=='man':
        clothes_path = './preprocessed/TL_man_clothes_2019.csv'
    elif gender=='woman':
        clothes_path = './preprocessed/TL_woman_clothes_2019.csv'
    else:
        print('Unknown arg: ',gender)
        return
    
    df_clothes = pd.read_csv(clothes_path)
    df_clothes = df_clothes[df_clothes['image'] == clothes_name]

    if len(df_clothes) == 0:
        print('bad clothes name: ',clothes_name)
        return
    return df_clothes.iloc[0:1,:]

def recommend_internal_season(clothes_path,rating_path,encoder_path,model_path,gender,age,color,face,body,season):
    
    reg = joblib.load(model_path)

    df_user=pd.DataFrame.from_dict({'r_gender':[gender],'age':[age],'personal_color':[color],'faceshape':[face],'bodyshape':[body]})
    df_clothes = pd.read_csv(clothes_path)
    df_clothes = df_clothes.loc[df_clothes['어울리는계절'] == season,:].reset_index(drop=True)
    #print(df_clothes.head())
    df_rating = pd.read_csv(rating_path,index_col='R_id').groupby('image').mean()[['선호여부']]

    df_rating.columns = ['평균선호']

    df = pd.concat([df_user,df_clothes],axis=1)
    df = df.ffill()
    df_clothes_name = df['image']
    df = df.drop(columns=['image'])

    encoder = ''
    with open(encoder_path,'rb') as f:
        encoder = pickle.load(f)
    df_encoded = encoder.transform(df.loc[:,'r_gender':'분위기'])
    df_encoded = pd.DataFrame(df_encoded,columns= [f"col{i}_{elem}" for i,sublist in enumerate(encoder.categories_) for elem in sublist])
    df_test = pd.concat([df_encoded,df.loc[:,'멋있다':].astype(np.int8)],axis=1)

    predict = reg.predict(df_test)
    predict_df = pd.DataFrame.from_dict({'예상평점':predict})
    #return pd.concat([df_clothes_name,rating],axis=1).sort_values(by=['rating'], axis=0, ascending=False).head(n=10)

    df_recommend = pd.concat([df_clothes_name,predict_df],axis=1).sort_values(by=['예상평점'], axis=0, ascending=False).head(n=100)
    df_recommend = pd.merge(df_recommend,df_rating,how='inner',on='image')

    df_recommend['종합평점'] = df_recommend['예상평점'] * 0.7 + df_recommend['평균선호'] * 0.3
    df_recommend = df_recommend.sort_values(by=['종합평점'], axis=0,ascending=False)
    return df_recommend

# season: summer,winter,spring
def recommend_season(gender,age,color,face,body,season):

    if season.lower() == 'summer':
        season = '여름'
    elif season.lower() == 'winter':
        season='겨울'
    else:
        season='봄/가을'

    if gender.lower() == 'man':
        gender='남성'
    elif gender.lower() == 'woman':
        gender='여성'
    else:
        print('Unknown arg : '+gender)
        return

    try:
        age = int(age) //10 * 10
        if age<20:
            age=20
        elif age>=60:
            age=50
        age=str(age)+'대'

    except Exception as e:
        print("Unknown arg : "+age)
        print("must be in 20~59")
        return

    color = color.lower()
    if color not in ['spring','summer','autumn','winter']:
        print("Unknown arg: "+color)
        return

    face = face.lower()
    if face not in ['heart','oblong','oval','round','square']:
        print("Unknown arg: "+face)
        return

    body = body.lower()
    if body not in ['trapezoid','round','rectangle','inverted_triangle','triangle','hourglass']:
        print("Unknown arg : "+body)
        return
    if (gender=='남성' and body=='hourglass') or (gender=='여성' and body=='trapezoid'):
        print('gender and bodyshape unmatch' + f'{gender},{body}')
        return
    try:
        if gender=='남성':
            df = recommend_internal_season('./preprocessed/TL_man_clothes_2019.csv','./preprocessed/TL_man_rating_2019.csv','./encoder/onehot_encoder_man.pkl','./model/random_man.pkl',gender,age,color,face,body,season)
            return df
        else:
            df = recommend_internal_season('./preprocessed/TL_woman_clothes_2019.csv','./preprocessed/TL_woman_rating_2019.csv','./encoder/onehot_encoder_woman.pkl','./model/random_woman.pkl',gender,age,color,face,body,season)
            return df
    except Exception as e:
        print(traceback.format_exc())
        return None


def recommend_internal(clothes_path,rating_path,encoder_path,model_path,gender,age,color,face,body):
    
    reg = joblib.load(model_path)

    df_user=pd.DataFrame.from_dict({'r_gender':[gender],'age':[age],'personal_color':[color],'faceshape':[face],'bodyshape':[body]})
    df_clothes = pd.read_csv(clothes_path)
    df_rating = pd.read_csv(rating_path,index_col='R_id').groupby('image').mean()[['선호여부']]

    df_rating.columns = ['평균선호']

    df = pd.concat([df_user,df_clothes],axis=1)
    df = df.ffill()
    df_clothes_name = df['image']
    df = df.drop(columns=['image'])

    encoder = ''
    with open(encoder_path,'rb') as f:
        encoder = pickle.load(f)
    df_encoded = encoder.transform(df.loc[:,'r_gender':'분위기'])
    df_encoded = pd.DataFrame(df_encoded,columns= [f"col{i}_{elem}" for i,sublist in enumerate(encoder.categories_) for elem in sublist])
    df_test = pd.concat([df_encoded,df.loc[:,'멋있다':].astype(np.int8)],axis=1)

    predict = reg.predict(df_test)
    predict_df = pd.DataFrame.from_dict({'예상평점':predict})
    #return pd.concat([df_clothes_name,rating],axis=1).sort_values(by=['rating'], axis=0, ascending=False).head(n=10)

    df_recommend = pd.concat([df_clothes_name,predict_df],axis=1).sort_values(by=['예상평점'], axis=0, ascending=False).head(n=100)
    df_recommend = pd.merge(df_recommend,df_rating,how='inner',on='image')

    df_recommend['종합평점'] = df_recommend['예상평점'] * 0.7 + df_recommend['평균선호'] * 0.3
    df_recommend = df_recommend.sort_values(by=['종합평점'], axis=0,ascending=False)
    return df_recommend


def recommend(gender,age,color,face,body):
    if gender.lower() == 'man':
        gender='남성'
    elif gender.lower() == 'woman':
        gender='여성'
    else:
        print('Unknown arg : '+gender)
        return

    try:
        age = int(age) //10 * 10
        if age<20:
            age=20
        elif age>=60:
            age=50
        age=str(age)+'대'

    except Exception as e:
        print("Unknown arg : "+age)
        print("must be in 20~59")
        return

    color = color.lower()
    if color not in ['spring','summer','autumn','winter']:
        print("Unknown arg: "+color)
        return

    face = face.lower()
    if face not in ['heart','oblong','oval','round','square']:
        print("Unknown arg: "+face)
        return

    body = body.lower()
    if body not in ['trapezoid','round','rectangle','inverted_triangle','triangle','hourglass']:
        print("Unknown arg : "+body)
        return
    if (gender=='남성' and body=='hourglass') or (gender=='여성' and body=='trapezoid'):
        print('gender and bodyshape unmatch' + f'{gender},{body}')
        return
    try:
        if gender=='남성':
            df = recommend_internal('./preprocessed/TL_man_clothes_2019.csv','./preprocessed/TL_man_rating_2019.csv','./encoder/onehot_encoder_man.pkl','./model/random_man.pkl',gender,age,color,face,body)
            return df
        else:
            df = recommend_internal('./preprocessed/TL_woman_clothes_2019.csv','./preprocessed/TL_woman_rating_2019.csv','./encoder/onehot_encoder_woman.pkl','./model/random_woman.pkl',gender,age,color,face,body)
            return df
    except Exception as e:

        return None
    
def update_model(gender,age,color,faceshape,bodyshape,clothes,rating):
    if gender =='man':
        gender='남성'
        clothes_path = './preprocessed/TL_man_clothes_2019.csv'
        model_path = './model/random_man.pkl'
        train_x_path = './train/train_x_man.csv'
        train_y_path = './train/train_y_man.csv'
        encoder_path = './encoder/onehot_encoder_man.pkl'
        rating_path = './preprocessed/TL_man_rating_2019.csv'
    else:
        gender='여성'
        clothes_path = './preprocessed/TL_woman_clothes_2019.csv'
        encoder_path = './encoder/onehot_encoder_woman.pkl'
        model_path = './model/random_woman.pkl'
        train_x_path = './train/train_x_woman.csv'
        train_y_path = './train/train_y_woman.csv'
        rating_path = './preprocessed/TL_woman_rating_2019.csv'
    
    try:
        ratings = pd.read_csv(rating_path)
        ratings= pd.concat([ratings,pd.DataFrame.from_dict({'R_id':[0],'image':[clothes],'선호여부':[rating],'스타일선호':[True]})],axis=0,ignore_index=True)
    
        ratings.to_csv(rating_path,index=False)

        train_x = pd.read_csv(train_x_path)
        train_y = pd.read_csv(train_y_path)

        df_user=pd.DataFrame.from_dict({'r_gender':[gender],'age':[str(age)+'대'],'personal_color':[color],'faceshape':[faceshape],'bodyshape':[bodyshape]})
        df_clothes = pd.read_csv(clothes_path)
        df_clothes = df_clothes[df_clothes['image']==clothes].reset_index(drop=True)
        df = pd.concat([df_user,df_clothes],axis=1)
    
        df = df.drop(columns=['image'])
        encoder = ''
        with open(encoder_path,'rb') as f:
            encoder = pickle.load(f)
        df_encoded = encoder.transform(df.loc[:,'r_gender':'분위기'])
        df_encoded = pd.DataFrame(df_encoded,columns= [f"col{i}_{elem}" for i,sublist in enumerate(encoder.categories_) for elem in sublist])
        df_append = pd.concat([df_encoded,df.loc[:,'멋있다':].astype(np.int8)],axis=1)

        train_x = pd.concat([train_x,df_append],axis=0,ignore_index=True)
        #print(train_x.tail())
        train_y = pd.concat([train_y,pd.DataFrame.from_dict({'선호여부':[rating]})],axis=0,ignore_index=True)['선호여부']
        #print(train_y.tail())

        train_x.to_csv(train_x_path,index=False)
        train_y.to_csv(train_y_path,index=False)

        reg = RandomForestRegressor(random_state=0,n_jobs=-1)
        reg.fit(train_x,train_y)
        joblib.dump(reg, model_path) 
    except:
        print('fail to update model...')

    