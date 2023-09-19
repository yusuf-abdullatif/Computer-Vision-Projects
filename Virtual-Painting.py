import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

myColors = [[5,93,0,179,255,255],
            [0,96,113,179,255,255],
            [0,88,139,179,255,255]]

def findColor(im, myColors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
    mask = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow(str(color[0]), mask)

while True:
    success, img = cap.read()
    findColor(img, myColors)
    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
