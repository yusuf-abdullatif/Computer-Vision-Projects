import cv2

faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
img = cv2.imread('Resources/lena.png')


cv2.imshow("Result", img)
cv2.waitKey(0)