import cv2
from deepface import DeepFace

# getting a haarcascade xml file and processing it
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# requesting input from camera
cap = cv2.VideoCapture(0)

# checking if we're getting video feed
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError('Can not open camera')

while True:
    # reading from video feed
    ret, frame = cap.read()

    # changing the video to grayscale to allow the face analysis to function properly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # draw a rectangle around the face in the image
    for (x, y, u, v) in faces:
        cv2.rectangle(frame, (x, y), (x+u, y+v), (0, 0, 225), 2)

    # we're using deepface's analyze class and 'frame' as input
    result = DeepFace.analyze(
        frame, actions=['emotion'], enforce_detection=False)

    # here we will only go print out the dominant emotion
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result['dominant_emotion'], (100, 100), font, 2,
                (225, 0, 0), 2, cv2.LINE_4)

    # this is the part where we display the output to the user
    cv2.imshow('Real time Emotion detection', frame)

    # here we specify the key that will stop the loop and all running processes
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
