import cv2
import mediapipe as mp
from rembg import remove
import numpy as np
from enum import Enum
# 어깨길이는 팔을 내려서, 가슴길이는 팔을 들어서 찍어야함.

HUMAN_THRESHOLD = 230
WAIST_RATIO = [[0.476127269,0.476127269,0.474836197,0.473165297,0.475631754],[0.47266639,0.481089448,0.479866888,0.479638317,0.476231349,0.475039778]]
#ratio는 https://sizekorea.kr/human-info/body-shape-class/age-gender-body?gender=M&age=20 에서 데이터를 가져와 비율 계산함
class Gender(Enum):
    MALE = 0
    FEMALE = 1 
class Age(Enum):
    TWENTY = 0
    THIRTY = 1
    FOURTY = 2
    FIFTY = 3
    SIXTY = 4
class BodyShape(Enum):
    HOURGLASS=0
    TRAPEZOID=1
    ROUND=2
    RECTANGLE=3
    INVERTED_TRIANGLE=4
    TRIANGLE=5
    


def getBodyShape_male(hip,shoulder,waist,bust,shoulder_height):
    threshold = int(shoulder_height * 0.01)
    if shoulder > hip+threshold and waist +threshold< hip  and hip+threshold<bust:
        return BodyShape.TRAPEZOID
    elif waist > shoulder+threshold and waist > hip+threshold:
        return BodyShape.ROUND
    elif shoulder > hip+threshold and shoulder > bust + threshold:
        return BodyShape.INVERTED_TRIANGLE
    elif hip > shoulder+threshold and hip > bust+threshold:
        return BodyShape.TRIANGLE
    return BodyShape.RECTANGLE
    

    
def getBodyShape_female(hip,shoulder,waist,bust,shoulder_height):
    threshold = int(shoulder_height * 0.01)
    if  waist+threshold < shoulder and waist+threshold < hip:
        return BodyShape.HOURGLASS
    elif waist > shoulder+threshold and waist > hip+threshold:
        return BodyShape.ROUND
    elif shoulder > hip+threshold and shoulder > bust+threshold:
        return BodyShape.INVERTED_TRIANGLE
    elif hip > shoulder+threshold and hip > bust+threshold:
        return BodyShape.TRIANGLE
    return BodyShape.RECTANGLE

def getBodyShape(image,image_handsup,gender,age):
    gender_dict={'man':0,'woman':1}
    age = age//10 * 10
    age_dict={20:0,30:1,40:2,50:3,60:4}
    


    gender_input = gender_dict[gender]
    age_input = age_dict[age]

    gender = Gender(gender_input)
    age = Age(age_input)

    # Mediapipe 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 이미지 로드
    
    if image is None:
        print('error : image is gone')
        return
    mask = remove(image,only_mask=True)
    image_height, image_width, _ = image.shape

    
    if image_handsup is None:
        print('error : image is gone')
        return
    mask_handsup = remove(image_handsup,only_mask=True)
    assert(image_height==image_handsup.shape[0] and image_width==image_handsup.shape[1])

    # Mediapipe Pose 사용
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:

        # BGR 이미지를 RGB로 변환
        results = pose.process(cv2.cvtColor(image_handsup, cv2.COLOR_BGR2RGB))
        results_for_shoulder = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 포즈 랜드마크가 검출되었는지 확인
        if results.pose_landmarks and results_for_shoulder.pose_landmarks:
            # 랜드마크 그리기
            #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 랜드마크 좌표 출력
            '''
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                print(f'Landmark {id}: ({x}, {y})')
            '''
            
            # detect shoulder and hip
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            # detect ankle
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            #detect foot index
            left_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

            #detect knee
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

            # 어깨 중간점 좌표 계산
            shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * image_width / 2 )
            shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * image_height / 2 )

            # 엉덩이 중간점 좌표 계산
            hip_center_x = int((left_hip.x + right_hip.x) * image_width / 2 )
            hip_center_y = int((left_hip.y + right_hip.y) * image_height / 2 )

            # 무릎 중간점 좌표 계산
            knee_center_x = int((left_knee.x + right_knee.x) * image_width / 2 )
            knee_center_y = int((left_knee.y + right_knee.y) * image_height / 2 )

            # 발목 중간점 좌표 계산
            ankle_center_x=int((left_ankle.x + right_ankle.x)*image_width/2)
            ankle_center_y=int((left_ankle.y+right_ankle.y)*image_height/2)

            # 발가락 중간점 좌표 계산
            foot_center_x = int((left_foot.x + right_foot.x)*image_width / 2)
            foot_center_y = int((left_foot.y + right_foot.y)*image_height / 2)



            #find hip length
            hiparr = mask_handsup[hip_center_y]
            hip_leftmost_index = np.where(hiparr[:hip_center_x] < HUMAN_THRESHOLD)[0]
            hip_rightmost_index = np.where(hiparr[hip_center_x:] < HUMAN_THRESHOLD)[0]

            hip_leftmost=0
            hip_rightmost=0

            
            if(hip_leftmost_index.size>0):
                hip_leftmost=hip_leftmost_index[-1]
            else:
                print('fail to determine hip_leftmost')
                return
            if(hip_rightmost_index.size>0):
                hip_rightmost=hip_rightmost_index[0] + hip_center_x
            else:
                print('fail to determine hip_rightmost')
                return
            

            #find waist length
            waist_center_x = (int)((hip_center_x + shoulder_center_x)/2)
            
            
            #from shoulder height, find waist height
            #허리높이 = (엉덩이높이+어깨높이) * waist_ratio (waist_ratio is variable by age and gender)
            shoulder_height = foot_center_y - shoulder_center_y
            hip_height = foot_center_y-hip_center_y
            waist_center_y = ((int)((shoulder_height+hip_height) * WAIST_RATIO[gender.value][age.value]) - foot_center_y)*-1
            
            waistarr = mask_handsup[waist_center_y]
            waist_leftmost_index = np.where(waistarr[:waist_center_x] < HUMAN_THRESHOLD)[0]
            waist_rightmost_index = np.where(waistarr[waist_center_x:] < HUMAN_THRESHOLD)[0]

            waist_leftmost = 0
            waist_rightmost = 0

            if (waist_leftmost_index.size>0):
                waist_leftmost = waist_leftmost_index[-1]
            else:
                print('fail to determine waist_leftmost')
                return
            if(waist_rightmost_index.size>0):
                waist_rightmost = waist_rightmost_index[0] + waist_center_x
            else:
                print('fail to determine waist_rightmost')
                return
            #find bust length
            bust_center_x = hip_center_x
            bust_center_y = (int)(shoulder_center_y + 0.12 * (ankle_center_y-shoulder_center_y))

            bustarr = mask_handsup[bust_center_y]
            bust_leftmost_index = np.where(bustarr[:bust_center_x] < HUMAN_THRESHOLD)[0]
            bust_rightmost_index = np.where(bustarr[bust_center_x:] <HUMAN_THRESHOLD)[0]

            bust_leftmost = 0
            bust_rightmost = 0

            if (bust_leftmost_index.size>0):
                bust_leftmost = bust_leftmost_index[-1]
            else:
                print('fail to determine bust_leftmost')
                return
            if(bust_rightmost_index.size>0):
                bust_rightmost = bust_rightmost_index[0] + bust_center_x
            else:
                print('fail to determine bust_rightmost')
                return
            
            #find shoulder length

            left_shoulder_2 = results_for_shoulder.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder_2 = results_for_shoulder.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_foot_2 = results_for_shoulder.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            right_foot_2 = results_for_shoulder.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            
            foot_center_x_2 = int((left_foot_2.x + right_foot_2.x)*image_width / 2)
            foot_center_y_2 = int((left_foot_2.y + right_foot_2.y)*image_height / 2)

            shoulder_center_x_2 = int((left_shoulder_2.x + right_shoulder_2.x) * image_width / 2 )
            shoulder_center_y_2 = int((left_shoulder_2.y + right_shoulder_2.y) * image_height / 2 )

            shoulder_height_2 = foot_center_y_2 - shoulder_center_y_2
            shoulderarr = mask[shoulder_center_y_2 - int(shoulder_height_2*0.03)]
            shoulder_leftmost_index = np.where(shoulderarr[:shoulder_center_x]< HUMAN_THRESHOLD)[0]
            shoulder_rightmost_index = np.where(shoulderarr[shoulder_center_x:] < HUMAN_THRESHOLD)[0]

            shoulder_leftmost = 0
            shoulder_rightmost = 0

            if(shoulder_leftmost_index.size>0):
                shoulder_leftmost = shoulder_leftmost_index[-1]
            else:
                print('fail to determine shoulder_leftmost')
                return
            if(shoulder_rightmost_index.size>0):
                shoulder_rightmost = shoulder_rightmost_index[0] + shoulder_center_x
            else:
                print('fail to determine shoulder_rightmost')
                return

            
            hip_len = hip_rightmost - hip_leftmost
            shoulder_len = shoulder_rightmost-shoulder_leftmost
            waist_len = waist_rightmost - waist_leftmost
            bust_len = bust_rightmost - bust_leftmost

            bodyshape=BodyShape.HOURGLASS
            #start bodyshape classification
            if gender == Gender.MALE:
                bodyshape = getBodyShape_male(hip_len,shoulder_len,waist_len,bust_len,abs(shoulder_height))
            else: #female
                bodyshape = getBodyShape_female(hip_len,shoulder_len,waist_len,bust_len,abs(shoulder_height))
            #end bodyshape classification
            #print('estimated bodyshape: ',bodyshape.name)
            return bodyshape.name.lower()
            '''
            cv2.line(image,(hip_leftmost,hip_center_y),(hip_rightmost,hip_center_y),(255,0,0),3)
            cv2.line(image,(shoulder_leftmost,shoulder_center_y),(shoulder_rightmost,shoulder_center_y),(0,255,0),3)
            cv2.line(image,(waist_leftmost,waist_center_y),(waist_rightmost,waist_center_y),(0,0,255),3)
            cv2.line(image,(bust_leftmost,bust_center_y),(bust_rightmost,bust_center_y),(255,0,255),3)

            cv2.line(image,(waist_leftmost,waist_center_y),(hip_leftmost,hip_center_y),(255,255,255),3)
            cv2.line(image,(waist_rightmost,waist_center_y),(hip_rightmost,hip_center_y),(255,255,255),3)

            cv2.line(image,(bust_leftmost,bust_center_y),(waist_leftmost,waist_center_y),(255,255,255),3)
            cv2.line(image,(bust_rightmost,bust_center_y),(waist_rightmost,waist_center_y),(255,255,255),3)

            
            #cv2.imshow('lines',image)
            cv2.imshow(str(bodyshape.name),cv2.resize(image, dsize=(0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
