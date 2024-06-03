import logging
import math
import threading
import time

import cv2
from ultralytics import YOLO

logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')


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

        self.tracked_objects = {}

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

    def track_objects(self, frame, positions):
        frame_height, frame_width, _ = frame.shape
        center_x, center_y = frame_width // 2, frame_height // 2

        for r in positions:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = math.ceil((box.conf[0] * 100)) / 100

                if confidence > 0.8:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    object_center_x = (x1 + x2) // 2
                    object_center_y = (y1 + y2) // 2

                    cv2.circle(frame, (object_center_x, object_center_y), 5, (0, 255, 0), -1)

                    text_center = f"Center: ({object_center_x}, {object_center_y})"
                    org_center = (x1, y2 + 40)

                    cv2.putText(frame, text_center, org_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    adjusted_center_x = x1 + (x2 - x1) // 2
                    adjusted_center_y = y1 + (y2 - y1) // 2

                    text_position = f"Position: ({x1}, {y1}) - ({x2}, {y2})"
                    org_position = (x1, y2 + 60)
                    cv2.putText(frame, text_position, org_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    text_adjusted_center = f"Adjusted Center: ({adjusted_center_x}, {adjusted_center_y})"
                    org_adjusted_center = (x1, y2 + 80)
                    cv2.putText(frame, text_adjusted_center, org_adjusted_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 1)

                    centroid_x = adjusted_center_x - center_x
                    centroid_y = center_y - adjusted_center_y

                    x, y = centroid_x, centroid_y

                    area_pixels = abs((x2 - x1) * (y2 - y1))

                    PIXEL_SIZE = 0.00001
                    area_real_world = area_pixels * PIXEL_SIZE

                    text_coords = f"X: {x:.2f}, Y: {y:.2f}"
                    text_remaining = f"Confidence: {int(confidence * 100)}%, Area: {area_real_world:.2f} m sq"

                    org_coords = (x1, y2 + 20)
                    org_remaining = (x1, y1 - 10)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    color = (0, 0, 0)
                    thickness = 1

                    cv2.putText(frame, text_coords, org_coords, font, fontScale, color, thickness)
                    cv2.putText(frame, text_remaining, org_remaining, font, fontScale, color, thickness)

                    # Log center coordinates along with other information
                    logging.info(
                        f"{text_coords}, {text_remaining}, {text_center}, {text_position}, {text_adjusted_center}")

        return frame


cameraProcess = Camera('./best.pt', 0)

while True:
    if cameraProcess.isCameraOpened:
        try:
            currentFrame = cameraProcess.getCurrentFrame()
            currentFrame = cv2.resize(currentFrame, (640, 640))
            objectsToDetect = ['red-circle', 'purple-circle', 'red-square']
            positions = cameraProcess.detectObjects(objectsToDetect, currentFrame)

            currentFrame = cameraProcess.track_objects(currentFrame, positions)

            cv2.imshow("Current Frame", currentFrame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                cameraProcess.isActive = False
                cameraProcess.cameraThread.join()
                break
            time.sleep(0.03)
        except Exception as e:
            print(e)
            time.sleep(3)

cv2.destroyAllWindows()
