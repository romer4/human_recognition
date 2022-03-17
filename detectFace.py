# import face_recognition
import cv2
import imutils
# import numpy as np

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")

# def detectFace(frame):
#     rgb_frame = frame[:, :, ::-1]#BGR to RGB
#     face_locations = face_recognition.face_locations(rgb_frame)
#     for x1, y1, x2, y2 in face_locations:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     cv2.imshow("Faces", frame)
#     return frame

def detectFace(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayFrame, 1.05, 5)
    print(faces)
    print("total faces: {}".format(len(faces)))
    for x1, y1, x2, y2 in faces:
        cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 1)
    cv2.imshow("Faces?", frame)

def detectFaceByPath(path):
    frame = cv2.imread(path)
    frame = imutils.resize(frame, width = min(500, frame.shape[1]))
    detectFace(frame)
    cv2.waitKey(0)
