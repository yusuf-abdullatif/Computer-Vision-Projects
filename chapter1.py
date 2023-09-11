

'''
def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640,240)
cv2.createTrackbar("Hue Min", "TrackBars", 3, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 139, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 73, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 109, 255, empty)


while True:
    img = cv2.imread("Resources/Biso.jpg")
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)

    cv2.waitKey(1)
'''




'''
 img = np.zeros((512, 512, 3), np.uint8)
 #print(img.shape)
 #img[:] = 255, 0, 0
 cv2.line(img, (0,0),(img.shape[1],img.shape[0]), (0,255,0),3)
 cv2.rectangle(img,(0,0),(250,350), (0,0,255),2)
 cv2.circle(img,(400,50),30,(255,255,0),5)
 cv2.putText(img," OPENCV ", (300,200), cv2.FONT_ITALIC, 1.5,(0,150,0),3)
cv2.imshow("Image", img)
cv2.waitKey(0)
'''



'''
img = cv2.imread("Resources/Biso.jpg")
print(img.shape)

imgResize = cv2.resize(img, (400,500))
print(imgResize.shape)

imgCropped = img[0:500, 200:500]

cv2.imshow("Image", img)
cv2.imshow("Image resize", imgResize)
cv2.imshow("Image Crop", imgCropped)
cv2.waitKey(0)
'''



'''
# Load an image from the specified file
img = cv2.imread("Resources/Biso.jpg")

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Convert the loaded image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)

# Detect edges in the original image using the Canny edge detector
imgCanny = cv2.Canny(img, 150, 200)

# Dilate the edges to make them thicker
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)

# Erode the dilated image to refine the edges
imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

# Display various image processing results using OpenCV's imshow function
cv2.imshow("Original Image", img)
cv2.imshow("Gray Image", imgGray)
cv2.imshow("Blurred Image", imgBlur)
cv2.imshow("Canny Edges", imgCanny)
cv2.imshow("Dilated Image", imgDilation)
cv2.imshow("Eroded Image", imgEroded)

# Wait until a key is pressed (0 means indefinitely)
cv2.waitKey(0)
'''