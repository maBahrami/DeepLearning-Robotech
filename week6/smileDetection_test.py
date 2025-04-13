import numpy as np
import cv2
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

detector = MTCNN()

smile_net = load_model(r"smileDetection_cnn.h5")

labels = ["not smile", "smile"]

def detect_faces(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = detector.detect_faces(rgb_img)[0]
    x, y, w, h = out["box"]

    return rgb_img[y:y+h, x:x+w], (x, y, w, h)

def preprocess(face):
    face = cv2.resize(face, (32, 32))
    face = face / 255.0
    face = np.array([face])
    
    return face



img = cv2.imread(r"week6\reference\1.jpg")
print(img.shape)

face, (x, y, w, h) = detect_faces(img)
normalized_face = preprocess(face)

out = smile_net.predict(normalized_face)[0]
max_index = np.argmax(out)
predict = labels[max_index]
probability = out[max_index]

text = f"{predict}: {probability}"

cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

