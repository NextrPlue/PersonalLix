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
    FORTY = 2
    FIFTY = 3
    SIXTY = 4

class BodyShape(Enum):
    HOURGLASS = 'hourglass'
    TRAPEZOID = 'trapezoid'
    ROUND = 'round'
    RECTANGLE = 'rectangle'
    INVERTED_TRIANGLE = 'inverted_triangle'
    TRIANGLE = 'triangle'

def get_body_shape(image, image_handsup, gender_str, age_int):
    gender_map = {'man': Gender.MALE, 'woman': Gender.FEMALE}
    age_map = {20: Age.TWENTY, 30: Age.THIRTY, 40: Age.FORTY, 50: Age.FIFTY, 60: Age.SIXTY}

    gender = gender_map.get(gender_str)
    age = age_map.get((age_int // 10) * 10, Age.TWENTY)

    if gender is None or age is None:
        return None

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(cv2.cvtColor(image_handsup, cv2.COLOR_BGR2RGB))
        results_shoulder = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks or not results_shoulder.pose_landmarks:
            return None

        image_height, image_width, _ = image.shape

        measurements = _compute_measurements(
            results.pose_landmarks.landmark,
            results_shoulder.pose_landmarks.landmark,
            image_width,
            image_height,
            gender.value,
            age.value,
            image,
            image_handsup
        )

        if not measurements:
            return None

        hip_len, shoulder_len, waist_len, bust_len, shoulder_height = measurements

        if gender == Gender.MALE:
            body_shape = _get_body_shape_male(hip_len, shoulder_len, waist_len, bust_len, shoulder_height)
        else:
            body_shape = _get_body_shape_female(hip_len, shoulder_len, waist_len, bust_len, shoulder_height)

        return body_shape.value

def _compute_measurements(landmarks, landmarks_shoulder, image_width, image_height, gender_value, age_value, image, image_handsup):
    mp_pose = mp.solutions.pose

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

    shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * image_width / 2)
    shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * image_height / 2)
    hip_center_x = int((left_hip.x + right_hip.x) * image_width / 2)
    hip_center_y = int((left_hip.y + right_hip.y) * image_height / 2)

    foot_center_x = int((left_foot.x + right_foot.x) * image_width / 2)
    foot_center_y = int((left_foot.y + right_foot.y) * image_height / 2)

    shoulder_height = foot_center_y - shoulder_center_y
    hip_height = foot_center_y - hip_center_y

    waist_ratio = WAIST_RATIO[gender_value][age_value]
    waist_center_y = int(foot_center_y - ((shoulder_height + hip_height) * waist_ratio))

    bust_center_y = int(shoulder_center_y + 0.12 * (foot_center_y - shoulder_center_y))
    bust_center_x = hip_center_x

    mask = remove(image, only_mask=True)
    mask_handsup = remove(image_handsup, only_mask=True)

    hip_len = _calculate_width(mask_handsup, hip_center_y, hip_center_x)
    waist_len = _calculate_width(mask_handsup, waist_center_y, hip_center_x)
    bust_len = _calculate_width(mask_handsup, bust_center_y, bust_center_x)

    shoulder_len = _calculate_shoulder_width(mask, landmarks_shoulder, image_width, image_height)

    if None in [hip_len, waist_len, bust_len, shoulder_len]:
        return None

    return hip_len, shoulder_len, waist_len, bust_len, abs(shoulder_height)

def _calculate_width(mask_image, center_y, center_x):
    row = mask_image[center_y]
    left_indices = np.where(row[:center_x] < HUMAN_THRESHOLD)[0]
    right_indices = np.where(row[center_x:] < HUMAN_THRESHOLD)[0]

    if left_indices.size > 0 and right_indices.size > 0:
        leftmost = left_indices[-1]
        rightmost = right_indices[0] + center_x
        width = rightmost - leftmost
        return width
    else:
        return None

def _calculate_shoulder_width(mask_image, landmarks_shoulder, image_width, image_height):
    mp_pose = mp.solutions.pose
    left_shoulder = landmarks_shoulder[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks_shoulder[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_foot = landmarks_shoulder[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    right_foot = landmarks_shoulder[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

    shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * image_width / 2)
    shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * image_height / 2)
    foot_center_y = int((left_foot.y + right_foot.y) * image_height / 2)
    shoulder_height = foot_center_y - shoulder_center_y

    measurement_y = shoulder_center_y - int(shoulder_height * 0.03)

    row = mask_image[measurement_y]
    left_indices = np.where(row[:shoulder_center_x] < HUMAN_THRESHOLD)[0]
    right_indices = np.where(row[shoulder_center_x:] < HUMAN_THRESHOLD)[0]

    if left_indices.size > 0 and right_indices.size > 0:
        leftmost = left_indices[-1]
        rightmost = right_indices[0] + shoulder_center_x
        width = rightmost - leftmost
        return width
    else:
        return None

def _get_body_shape_male(hip, shoulder, waist, bust, shoulder_height):
    threshold = int(shoulder_height * 0.01)
    if shoulder > hip + threshold and waist + threshold < hip and hip + threshold < bust:
        return BodyShape.TRAPEZOID
    elif waist > shoulder + threshold and waist > hip + threshold:
        return BodyShape.ROUND
    elif shoulder > hip + threshold and shoulder > bust + threshold:
        return BodyShape.INVERTED_TRIANGLE
    elif hip > shoulder + threshold and hip > bust + threshold:
        return BodyShape.TRIANGLE
    else:
        return BodyShape.RECTANGLE

def _get_body_shape_female(hip, shoulder, waist, bust, shoulder_height):
    threshold = int(shoulder_height * 0.01)
    if waist + threshold < shoulder and waist + threshold < hip:
        return BodyShape.HOURGLASS
    elif waist > shoulder + threshold and waist > hip + threshold:
        return BodyShape.ROUND
    elif shoulder > hip + threshold and shoulder > bust + threshold:
        return BodyShape.INVERTED_TRIANGLE
    elif hip > shoulder + threshold and hip > bust + threshold:
        return BodyShape.TRIANGLE
    else:
        return BodyShape.RECTANGLE
