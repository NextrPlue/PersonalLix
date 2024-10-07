import tensorflow as tf
import cv2
import numpy as np
interpreter = tf.lite.Interpreter(model_path='model/faceshape_efficientnetb4_crop.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

faceshape = ['heart','oblong','oval','round','square']
face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
def get_face_shape(faceimage):
    
    
    try:
        gray = cv2.cvtColor(faceimage,cv2.COLOR_BGR2GRAY)
    except:
        return None
    faces = face_cascade.detectMultiScale(gray)
    if len(faces)>0:
        x,y,w,h = faces[0]
    else:
        return None

    img = cv2.cvtColor(faceimage,cv2.COLOR_BGR2RGB)
    img = img[y:y+h,x:x+w]
    img = cv2.resize(img,(380,380))

    arr = np.array(img)

    mean = [159.80679614507162,122.67351405273274,104.61857584263046]
    std = [72.58244780862275,62.41943811258287,59.047168710327774]

    for i in range(3):
        arr[:,:,i] = (arr[:,:,i].astype(np.float32) - mean[i])/std[i]
    
    arr = np.expand_dims(arr,axis=0)
    interpreter.set_tensor(input_details[0]['index'], arr.astype(np.float32))

    # 추론 실행
    interpreter.invoke()

    # 출력 텐서에서 결과 가져오기
    result = interpreter.get_tensor(output_details[0]['index'])
    #print("'Heart': 0, 'Oblong': 1, 'Oval': 2, 'Round': 3,'Square':4")
    #print(result[0])
    return faceshape[result[0].argmax()]
