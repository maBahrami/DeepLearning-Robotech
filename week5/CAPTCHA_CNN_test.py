import cv2
from tensorflow.keras.models import load_model
import numpy as np

net = load_model(r"CAPTCHA_cnn.h5")

img = cv2.imread(r"Week5\reference\captcha_sample4.PNG")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    roi = img[y-5:y+h+5, x-5:x+w+5]
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi = cv2.resize(roi, (32, 32))
    roi = roi / 255
    roi = np.array([roi])

    out = net.predict(roi)[0]
    max_index = np.argmax(out) + 1
    print(max_index)


    cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2)
    cv2.putText(img, str(max_index), (x+2, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 255), 1)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()