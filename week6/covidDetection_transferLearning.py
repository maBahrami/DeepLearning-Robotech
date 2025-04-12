import cv2
import glob
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt


data = []
labels = []


for item in glob.glob(r"week6\reference\covid19-dataset\dataset\*\*"):

    img = cv2.imread(item)
    print(item)

    r_img = cv2.resize(img, (224, 224))
    data.append(r_img)

    label = item.split("\\")[-2]
    labels.append(label)


le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

data = np.array(data) / 255

x_train, x_test, y_train, y_test = train_test_split(data, labels, 
                                                    test_size=0.15, random_state=42)

aug = ImageDataGenerator(rotation_range = 10,
                         fill_mode = "nearest")

baseModel = VGG16(weights = "imagenet",
                  include_top = False,
                  input_tensor = layers.Input(shape = (224, 224, 3)))


for layer in baseModel.layers:
    layer.trainable = False


Network = models.Sequential([
                            baseModel,
                            layers.MaxPooling2D((4, 4)),
                            layers.Flatten(),
                            layers.Dense(64, activation = "relu"),
                            #layers.Dropout(0.5),
                            layers.Dense(2, activation = "softmax")
                            ])

opt = Adam(learning_rate = 0.001, decay = 0.001/25)

Network.compile(optimizer = opt,
                loss = "binary_crossentropy",
                metrics = ["accuracy"])

H = Network.fit(aug.flow(x_train, y_train, batch_size = 8),
                          steps_per_epoch = len(x_train) // 8,
                          validation_data = (x_test, y_test),
                          epochs = 25
                          )


plt.style.use("ggplot")
plt.plot(np.arange(25), H.history["accuracy"], label = "acc")
plt.plot(np.arange(25), H.history["val_accuracy"], label = "val_acc")
plt.plot(np.arange(25), H.history["loss"], label = "loss")
plt.plot(np.arange(25), H.history["val_loss"], label = "val_loss")
plt.xlabel("epochs")
plt.ylabel("accuracy/loss")
plt.legend()
plt.show()

