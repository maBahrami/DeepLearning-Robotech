import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from joblib import dump
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

data = []
labels = []

for i, item in enumerate(glob.glob(r"C:\Users\mabah\Desktop\ML_Robotech\Week7\HW\reference\Q3\train\*\*")):
    img = cv2.imread(item)
  
    img = cv2.resize(img, (32, 32))

    img = img / 255
    data.append(img)

    label = item.split("\\")[-1].split(".")[0]
    labels.append(label)

    if i % 100 == 0:
        print(f"[INFO]  {i}/2000 processed")

data = np.array(data)
labels = np.array(labels).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1234)

enc = OneHotEncoder(sparse_output=False)
y_train = enc.fit_transform(y_train)
y_test = enc.transform(y_test)


EPOCHS = 45
BATCH_SIZE = 8

net = models.Sequential([
                        layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
                        layers.MaxPool2D((2, 2)),
                        layers.Conv2D(32, (3, 3), activation = 'relu'),
                        layers.MaxPool2D((2, 2)),
                        layers.Flatten(),
                        layers.Dense(100, activation="relu"),
                        layers.Dense(2, activation="softmax")
                        ])

#net.summary()

net.compile(optimizer="SGD",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

H = net.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

loss, acc = net.evaluate(x_test, y_test)
print(f"loss: {loss}, accuaracy{acc}")

net.save("CatDog_CNN.h5")


plt.style.use("ggplot")
plt.plot(np.arange(EPOCHS), H.history["loss"], label="train loss")
plt.plot(np.arange(EPOCHS), H.history["val_loss"], label="test loss")
plt.plot(np.arange(EPOCHS), H.history["accuracy"], label="train accuracy")
plt.plot(np.arange(EPOCHS), H.history["val_accuracy"], label="test accuracy")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("training on cat/dog dataset")
plt.show()