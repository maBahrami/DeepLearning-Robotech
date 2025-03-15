import cv2
import glob
from joblib import load
import numpy as np
from tensorflow.keras import models 

clf = models.load_model(r"fireDetection_CNN.h5")

output_label = ["fire", "non fire"]

for item in glob.glob(r"C:\Users\mabah\Desktop\ML_Robotech\Week5\reference\Datasets\fire_test\*"):
    img = cv2.imread(item)
    r_img = cv2.resize(img, (32, 32))
    r_img = r_img / 255
    #r_img = r_img.flatten()

    out = clf.predict(np.array([r_img]))[0]
    max_pred = np.argmax(out)
    out = output_label[max_pred]
    print(type(out))
    print(out)

    cv2.putText(img, out, (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("image", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
