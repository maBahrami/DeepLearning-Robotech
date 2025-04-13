import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from mtcnn import MTCNN
import cv2


detector = MTCNN()

cap = cv2.VideoCapture(r"week6\reference\video1.mp4")

while True:
    _, img = cap.read()
    #img = cv2.resize(img, (640/2, 360/2))

    if img is None: break

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    try:
        out = detector.detect_faces(rgb_img)[0]

        x, y, w, h = out["box"]

        confidence = round(out["confidence"], 2)
        text = f"prob: {confidence*100}"

        kp = out["keypoints"]
        for _, value in kp.items():
            cv2.circle(img, value, 5, (0, 0, 255), -1)


        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("image", img)
        if cv2.waitKey(30) == ord("q"): break

    except:
        pass

cv2.destroyAllWindows()



