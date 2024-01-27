import cv2
import numpy as np

url = "http://10.151.63.69:8080/video"
cap = cv2.VideoCapture(url)

while True:
    ret, img = cap.read()

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for white color
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find the contours of white area
    contour_white, _ = cv2.findContours(white_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of the image
    center_image = np.array(img.shape[1::-1]) // 2

    # Find the contour with the minimum distance to the center
    min_contour = None
    min_distance = float('inf')
    for contour in contour_white:
        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Calculate the Euclidean distance to the center of the image
            distance = np.linalg.norm(center_image - np.array([cX, cY]))
            # Check if this contour is closer than the previous minimum
            if distance < min_distance:
                min_distance = distance
                min_contour = contour

    # If a contour is found, draw a bounding box around it
    if min_contour is not None:
        x, y, w, h = cv2.boundingRect(min_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Threshold the image
    ret, threshold_img = cv2.threshold(white_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(threshold_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # Display the original image in color
    cv2.imshow("Color Image", img)

    # Display the thresholded image
    cv2.imshow("Threshold Image", threshold_img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
