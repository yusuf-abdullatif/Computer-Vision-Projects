# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

cam_id = 2

# Lists available webcams on system
def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

# Callback to handle mouse click to extract color at cursor position
def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = frame[y,x,0]
        colorsG = frame[y,x,1]
        colorsR = frame[y,x,2]
        colors = frame[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)

def simple_webcam():
    global frame
    # Open webcam
    cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    cv2.namedWindow('my webcam')
    cv2.setMouseCallback('my webcam', mouseRGB)

    grayscale = False
    lowpass = False
    sobel = False
    sobel_mode = 1
    kx = 21
    ky = 21
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    while True:
        ret_val, img = cam.read()
        frame = img.copy()

        if lowpass:
            img = cv2.GaussianBlur(img,(kx,ky), 0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if grayscale:
            img = gray

        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        if sobel:
            if sobel_mode == 1:
                img = abs_grad_x
            if sobel_mode == 2:
                img = abs_grad_y
            if sobel_mode == 3:
                img = grad

        cv2.imshow('my webcam', img)
        ch = cv2.waitKey(1)
        if ch == 27:
            break  # esc to quit
        if ch == ord('l'):
            lowpass = not lowpass
        if ch == ord('g'):
            grayscale = not grayscale
        if ch == ord('s'):
            sobel = not sobel
        if ch == ord('1'):
            sobel_mode = 1
        if ch == ord('2'):
            sobel_mode = 2
        if ch == ord('3'):
            sobel_mode = 3

    cv2.destroyAllWindows()


def orb_test():
    # Open webcam
    cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    # Create ORB Feature extractor
    orb = cv2.ORB_create()

    while True:
        ret_val, img = cam.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the keypoints with ORB
        kp = orb.detect(img, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        # draw only keypoints location,not size and orientation
        img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

def face_test():
    # Open webcam
    cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        # Capture new image frame
        ret_val, img = cam.read()

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the output
        cv2.imshow('Faces', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
#    list_ports()
    simple_webcam()
    orb_test()
    face_test()
