import cv2
import numpy as np

img = cv2.imread(r"week4\reference\square.PNG", 0)
img = cv2.imread(r"week4\reference\test2.jpg", 0)



kernel = np.array([[1, -1]])
kernel2 = np.array([[1],
                    [-1]])

conv_img = cv2.filter2D(img, cv2.CV_8U, kernel)
conv_img2 = cv2.filter2D(img, cv2.CV_8U, kernel2)

cv2.imshow("image", img)
cv2.imshow("conv image", conv_img)
cv2.imshow("conv image2", conv_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()