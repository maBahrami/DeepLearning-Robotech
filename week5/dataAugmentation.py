from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

img = cv2.imread(r"week5\reference\MontBlanc.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array([img])

aug = ImageDataGenerator(rotation_range = 30,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         shear_range = 0.2,
                         zoom_range = 0.2,
                         horizontal_flip = True,
                         fill_mode = "reflect")

imageGen = aug.flow(img, batch_size=1, save_to_dir=r"week5\reference\generated", save_format="jpg", save_prefix="gen")


counter = 0
for image in imageGen:
    counter += 1

    if counter == 10: break 