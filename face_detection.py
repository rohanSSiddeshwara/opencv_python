# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import matplotlib.pyplot as plt
import sys
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
image=cv2.imread("images.jpg")





def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    cv2.imshow("image", image)
    #just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=20);
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_copy

	# Display the resulting frame
cv2.imshow("Faces found", detect_faces(faceCascade,image))
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
else:
    sys.exit()
