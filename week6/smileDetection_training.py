import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mtcnn import MTCNN
import cv2
import numpy as np
import glob
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
plt.style.use("ggplot")

detector = MTCNN()

def detect_faces(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = detector.detect_faces(rgb_img)[0]
    x, y, w, h = out["box"]

    return rgb_img[y:y+h, x:x+w]

all_faces = []
all_labels = []

for i, item in enumerate(glob.glob(r"week6\reference\smile_dataset\*\*")):

    img = cv2.imread(item)
    try:
        face = detect_faces(img)

        face = cv2.resize(face, (32, 32))
        face = face / 255.0

        all_faces.append(face)

        label = item.split("\\")[-2]
        print(label)
        all_labels.append(label)
    except:
        pass

    if i % 100 == 0:
        print(f"[INFO] {i}/4000 processed")


all_faces = np.array(all_faces)

le = LabelEncoder()
all_labels_le = le.fit_transform(all_labels)
all_labels_le = to_categorical(all_labels_le)

trainX, testX, trainY, testY = train_test_split(all_faces, all_labels_le, test_size=0.2)

net = models.Sequential([
                            layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
                            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                            layers.MaxPooling2D((2, 2)),

                            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                            layers.MaxPooling2D((2, 2)),

                            layers.Flatten(),
                            layers.Dense(32, activation="relu"),
                            layers.Dense(2, activation="softmax")
                        ])

net.compile(optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

H = net.fit(trainX, trainY, batch_size=32, epochs=25, validation_data=(testX, testY))

net.save("smileDetection_cnn.h5")

print(net.summary())


plt.plot(H.history["accuracy"], label = "train accuracy")
plt.plot(H.history["val_accuracy"], label = "test accuracy")
plt.plot(H.history["loss"], label = "train loss")
plt.plot(H.history["val_loss"], label = "test loss")
plt.xlabel("epochs")
plt.ylabel("accuracy/loss")
plt.legend()
plt.show()


