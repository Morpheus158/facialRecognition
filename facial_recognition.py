from cv2 import cv2
import sys
import time

cascPath = r"C:\Users\Pavel\AppData\Roaming\Python\Python38\site-packages\cv2\data\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Video", gray)

    cv2.waitKey(1)

video.release()
if cv2.waitKey(25) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

"""
# Get user supplied values
imagePath = "photo.jpg"
"""
#cascPath = r"C:\Users\Pavel\AppData\Roaming\Python\Python38\site-packages\cv2\data\haarcascade_frontalface_default.xml"
"""
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
"""