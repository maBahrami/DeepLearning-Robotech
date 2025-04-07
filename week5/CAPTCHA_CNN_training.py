import cv2
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


def load_data():

    data = []
    labels = []

    for i, item in enumerate(glob.glob(r"week5\reference\CAPTCHA\*\*")):
        img = cv2.imread(item)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))
        img = img / 255

        data.append(img)

        label = item.split("\\")[-2]
        labels.append(label)


        if i % 100 == 0: print(f"[INFO] {i}/2011 processed")

    data = np.array(data)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1234)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, x_test, y_train, y_test



def training():
    net = models.Sequential([
                            layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
                            layers.MaxPooling2D((2, 2)),
                            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                            layers.MaxPooling2D((2, 2)),
                            layers.Flatten(),
                            layers.Dense(32, activation="relu"),
                            layers.Dense(9, activation="softmax")
                            ])

    net.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    H = net.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

    net.save("CAPTCHA_cnn.h5")
    
    print(net.summary())

    return H



def show_plots():
    plt.style.use("ggplot")
    plt.plot(H.history["accuracy"], label = "train accuracy")
    plt.plot(H.history["val_accuracy"], label = "test accuracy")
    plt.plot(H.history["loss"], label = "train loss")
    plt.plot(H.history["val_loss"], label = "test loss")
    plt.xlabel("epochs")
    plt.ylabel("accuracy/loss")
    plt.legend()
    plt.show()




x_train, x_test, y_train, y_test = load_data()

H = training()

show_plots()





