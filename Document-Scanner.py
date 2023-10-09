import cv2

########################
widthImg=640
heightImg=380
########################



frameWidth = widthImg
frameHeight= heightImg
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBLur = cv2.GaussianBlur(imgGray, (5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)



while True:
    success, img = cap.read()
    cv2.imshow("Result",img)
    img = cv2.resize(img,(widthImg, heightImg))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

