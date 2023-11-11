import cv2
import time
import threading
#
import math
from ultralytics import YOLO

class Camera:
    def __init__(self, modelFile, videoCapturePort):
        self.modelFile = modelFile
        self.videoCapturePort = videoCapturePort
        self.camera = self.openCamera()
        self.currentFrame = None
        self.isActive = True
        self.isCameraOpened = False
        self.cameraThread = threading.Thread(target=self.readCamera)
        self.model = YOLO(self.modelFile)

        self.cameraThread.start()
        while not self.isCameraOpened:
            self.isCameraOpened = self.camera.isOpened()

    def openCamera(self):
        return cv2.VideoCapture(self.videoCapturePort)

    def readCamera(self):
        while self.isActive:
            ret, frame = self.camera.read()
            self.currentFrame = frame

        self.camera.release()

    def getCurrentFrame(self):
        return self.currentFrame

    def detectObjects(self, objectsToDetect, frame):
        results = self.model(frame, stream=True)
        return results


cameraProcess = Camera('./ultralytics/duck.pt', 0)

while True:
    if cameraProcess.isCameraOpened:
        try:
            currentFrame = cameraProcess.getCurrentFrame()
            objectsToDetect = ['duck']
            positions = cameraProcess.detectObjects(objectsToDetect, currentFrame)

            # print(positions)
            for r in positions:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                    # confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    if confidence > 0.8:

                        cv2.rectangle(currentFrame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        cls = int(box.cls[0])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(currentFrame, str(confidence), org, font, fontScale, color, thickness)

            cv2.imshow("Current Frame", currentFrame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                cameraProcess.isActive = False
                cameraProcess.cameraThread.join()
                quit()
            time.sleep(0.03)
        except Exception as e:
            print(e)
            time.sleep(3)