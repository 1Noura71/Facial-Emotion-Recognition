import cv2
from deepface import DeepFace

# loading image
img = cv2.imread('img/angry_face.jpg')

# getting a haarcascade xml file and processing it
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# changing the video to grayscale to allow the face analysis to function properly
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# draw a rectangle around the face in the image
for (x, y, u, v) in faces:
    cv2.rectangle(img, (x, y), (x+u, y+v), (0, 0, 225), 2)

# we're using deepface's analyze class and 'frame' as input
result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

# here we will only go print out the dominant emotion
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, result['dominant_emotion'], (100, 100), font, 2,
            (225, 0, 0), 2, cv2.LINE_4)

# this is the part where we display the output to the user
cv2.imshow('image', img)

# press any key to close the window
cv2.waitKey(0)

cv2.destroyAllWindows()
