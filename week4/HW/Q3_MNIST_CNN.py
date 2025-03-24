import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


path = r"C:\Users\mabah\Desktop\DL_Robotech\week4\HW\reference\mnist.npz"

with np.load(path, allow_pickle=True) as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

# Print shape to verify
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# reshape for channels
#x_train = x_train.reshape(-1, 28, 28, 1)
#x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# Define a simple CNN
model = Sequential([
                        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                        MaxPooling2D((2, 2)),
                        Conv2D(64, (3, 3), activation="relu"),
                        Flatten(),
                        Dense(10, activation="softmax")  
                    ])
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

EPOCHS = 3
H = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print(f"loss: {loss}, accuaracy{acc}")


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