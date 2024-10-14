import cv2
import numpy as np
import joblib
import dlib

kmeans_model = joblib.load('models/model_files/kmeans_model_L2.pkl')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/model_files/shape_predictor_68_face_landmarks.dat')

def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)
    landmark_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))
    return np.array(landmark_points, np.int32)

def create_skin_mask(image, landmarks):
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))
    eyebrow_indices = list(range(17, 27))
    mouth_indices = list(range(48, 61))

    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    cv2.fillPoly(mask, [landmarks[left_eye_indices]], 0)
    cv2.fillPoly(mask, [landmarks[right_eye_indices]], 0)
    cv2.fillPoly(mask, [landmarks[eyebrow_indices]], 0)
    cv2.fillPoly(mask, [landmarks[mouth_indices]], 0)

    face_indices = list(range(0, 17)) + list(range(17, 27)) + list(range(27, 36)) + list(range(48, 68))
    face_hull = cv2.convexHull(landmarks[face_indices])

    face_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(face_mask, face_hull, 255)

    skin_mask = cv2.bitwise_and(mask, face_mask)
    return skin_mask

def extract_skin_rgb(image, skin_mask):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img[skin_mask > 0]

def calculate_quartile_means(rgb_codes):
    r = np.sort(rgb_codes[:, 0])
    g = np.sort(rgb_codes[:, 1])
    b = np.sort(rgb_codes[:, 2])

    r_filtered = r[(r >= np.percentile(r, 25)) & (r <= np.percentile(r, 75))]
    g_filtered = g[(g >= np.percentile(g, 25)) & (g <= np.percentile(g, 75))]
    b_filtered = b[(b >= np.percentile(b, 25)) & (b <= np.percentile(b, 75))]

    return np.mean(r_filtered), np.mean(g_filtered), np.mean(b_filtered)

def rgb_to_hsv_and_lab(mean_r, mean_g, mean_b):
    rgb_pixel = np.uint8([[[mean_r, mean_g, mean_b]]])
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2Lab)[0][0]
    return hsv_pixel[2], hsv_pixel[1], lab_pixel[2]  # V, S, b

def predict_cluster(kmeans_model, vsb_values):
    return kmeans_model.predict([vsb_values])[0]

def cluster_to_season(cluster):
    seasons = {0: "spring", 1: "summer", 2: "autumn", 3: "winter"}
    return seasons.get(cluster, "unknown")

def get_personal_color(image):
    landmarks = get_face_landmarks(image)
    if landmarks is None:
        return None
    skin_mask = create_skin_mask(image, landmarks)
    skin_rgb_codes = extract_skin_rgb(image, skin_mask)
    if skin_rgb_codes.size == 0:
        return None
    mean_r, mean_g, mean_b = calculate_quartile_means(skin_rgb_codes)
    v, s, b = rgb_to_hsv_and_lab(mean_r, mean_g, mean_b)
    cluster = predict_cluster(kmeans_model, [v, s, b])
    season = cluster_to_season(cluster)
    return season
