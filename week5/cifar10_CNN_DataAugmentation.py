import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer



def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(y_train)
    x_train, x_test = x_train/255.0, x_test/255.0

    le = LabelBinarizer()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return x_train, x_test, y_train, y_test


def training():
    aug = ImageDataGenerator(rotation_range = 30,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         shear_range = 0.2,
                         zoom_range = 0.2,
                         horizontal_flip = True,
                         fill_mode = "reflect")
    
    net = models.Sequential([
                                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
                                layers.BatchNormalization(),
                                layers.Conv2D(32, (3, 3), activation="relu"),
                                layers.BatchNormalization(),
                                layers.MaxPool2D(),

                                layers.Conv2D(64, (3, 3), activation="relu"),
                                layers.BatchNormalization(),
                                layers.Conv2D(64, (3, 3), activation="relu"),
                                layers.BatchNormalization(),
                                layers.MaxPool2D(),                               

                                layers.Flatten(),
                                layers.Dense(512, activation="relu"),
                                layers.BatchNormalization(),
                                layers.Dense(10, activation="softmax")
                            ])

    net.compile(optimizer = "SGD",
                metrics = ["accuracy"],
                loss = "categorical_crossentropy")

    H = net.fit(aug.flow(x_train, y_train, batch_size=32),
                steps_per_epoch=len(x_train)//32,
                validation_data=(x_test, y_test), 
                epochs=15)

    return H

def show_results(H):
    plt.plot(H.history["accuracy"], label="train accuracy")
    plt.plot(H.history["val_accuracy"], label="test accuracy")
    plt.plot(H.history["loss"], label="train loss")
    plt.plot(H.history["val_loss"], label="test loss")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy/loss")
    plt.title("cifar classifier")
    plt.legend()
    plt.show()




x_train, x_test, y_train, y_test = load_data()

H = training()

show_results(H)