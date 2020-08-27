import cv2
import os

# credits: Shantnu Tiwari 
# adapted from https://realpython.com/face-detection-in-python-using-a-webcam/

casc_file = os.path.join('..', 'cascade', 'hand.xml')
vid_file = 0
faceCascade = cv2.CascadeClassifier(casc_file)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hands = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.12,
        minNeighbors=5,
        minSize=(32, 32)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(30) & 0xff == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
