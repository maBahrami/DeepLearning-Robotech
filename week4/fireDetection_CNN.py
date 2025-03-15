import cv2
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras import models

EPOCHS = 20
batchSize = 16

#Week5\reference\Datasets\fire_dataset
#C:\Users\mabah\Desktop\ML_Robotech\Week5\reference\Datasets\fire_dataset

def load_data():

    data_list = []
    labels = []

    le = LabelEncoder()

    for i, address in enumerate(glob.glob(r"C:\Users\mabah\Desktop\ML_Robotech\Week5\reference\Datasets\fire_dataset\*\*")):
        img = cv2.imread(address)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        data_list.append(img)

        label = address.split("\\")[-1].split(".")[0]
        labels.append(label)

        if i % 100 == 0:
            print(f"[INFO] {i}/{1000} processed.")


    data_list = np.array(data_list)

    x_train, x_test, y_train, y_test = train_test_split(data_list, labels, test_size=0.2, random_state=123)

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    return x_train, x_test, y_train, y_test


def convolutional_neural_network():
    net = models.Sequential([
                            layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
                            layers.MaxPool2D((2, 2)),
                            layers.Conv2D(32, (3, 3), activation = 'relu'),
                            layers.MaxPool2D((2, 2)),
                            layers.Flatten(),
                            layers.Dense(100, activation="relu"),
                            layers.Dense(2, activation="sigmoid")
                            ])

    net.summary()

    net.compile(optimizer="SGD",
                loss="binary_crossentropy",
                metrics=["accuracy"])

    H = net.fit(x_train, y_train, batch_size=batchSize, epochs=EPOCHS, validation_data=(x_test, y_test))

    loss, acc = net.evaluate(x_test, y_test)
    print(f"loss: {loss}, accuaracy{acc}")

    net.save("fireDetection_CNN.h5")

    # net = models.load_model("fire_detection_NeuralNetwork.h5")

    return H

def show_result():

    print(H.history.keys())

    plt.style.use("ggplot")
    plt.plot(np.arange(EPOCHS), H.history["loss"], label="train loss")
    plt.plot(np.arange(EPOCHS), H.history["val_loss"], label="test loss")
    plt.plot(np.arange(EPOCHS), H.history["accuracy"], label="train accuracy")
    plt.plot(np.arange(EPOCHS), H.history["val_accuracy"], label="test accuracy")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("training on fire dataset")
    plt.show()


x_train, x_test, y_train, y_test = load_data()

H = convolutional_neural_network()

show_result()
