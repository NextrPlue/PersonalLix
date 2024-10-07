import tensorflow as tf
import cv2
import numpy as np

interpreter = tf.lite.Interpreter(model_path='models/model_files/faceshape_efficientnetb4_crop.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

FACES_SHAPES = ['heart','oblong','oval','round','square']
face_cascade = cv2.CascadeClassifier('models/model_files/haarcascade_frontalface_default.xml')

def get_face_shape(face_image):
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return None

    faces = face_cascade.detectMultiScale(gray)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]

    img_rgb = cv2.cvtColor(face_image,cv2.COLOR_BGR2RGB)
    face_crop = img_rgb[y:y+h,x:x+w]
    face_resized = cv2.resize(face_crop,(380,380))

    img_array = np.array(face_resized)
    mean = [159.80679614507162,122.67351405273274,104.61857584263046]
    std = [72.58244780862275,62.41943811258287,59.047168710327774]
    img_normalized = (img_array - mean) / std
    img_expanded = np.expand_dims(img_normalized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_expanded.astype(np.float32))
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    face_shape_index = np.argmax(result[0])
    return FACES_SHAPES[face_shape_index]
