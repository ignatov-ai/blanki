# импортирование необходимых библиотек
import numpy as np
import cv2
import imutils

def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 1)
    return new_img

# параметр для сканируемого изображения
args_image = 'blanki/02.jpg'

# прочитать изображение
img_original = cv2.imread(args_image)
height, width = img_original.shape[:2]
new_size = width // 3, height // 3
image = cv2.resize(img_original,new_size)
cv2.imshow("Original image", image)
scanBW = image.copy()

cv2.imshow("Rotated image", scanBW)
cv2.waitKey(1000)
cv2.destroyWindow("Rotated image")

height, width = scanBW.shape[:2]
print(width,height)

for i in range(3):
    scanBW = RotateClockWise90(scanBW)
    cv2.imshow("Rotated image", scanBW)
    cv2.waitKey(1000)
    cv2.destroyWindow("Rotated image")

cv2.waitKey(0)