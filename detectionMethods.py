import cv2
import imutils
from filters import *
from log import *

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")

def detectFace(frame, known_image=None, use_filter=False):
    if use_filter:
        print("O método detectFace está usando um filtro de rosto!")

    faces = face_cascade.detectMultiScale(frame, 1.05, 5)
    for x1, y1, x2, y2 in faces:
        if use_filter:
            unknown_image = frame[y1:y1 + y2, x1:x1 + x2]
            comparation = compareFaces(known_image, unknown_image)
            cv2.imshow("{}Accuracy: {}".format(x1, str(comparation[1])), unknown_image)
            log("Accuracy: {}%".format(str(comparation[1])))
            if comparation[0] == False: continue

        cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 1)
    cv2.imshow("Faces?", frame)
    return frame

def detectObjects(frame):
    with open("./datasets/coco.names", "rt") as file:
        CLASSES = file.read().rstrip("\n").split("\n")

    net = cv2.dnn_DetectionModel(r"./datasets/frozen_inference_graph.pb", r"./datasets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    indexes, confidences, coords = net.detect(frame, confThreshold=0.5)
    if len(indexes) > 0:
        for index, confidence, coord in zip(indexes.flatten(), confidences.flatten(), coords):
            label = "{}: {:.2f}%".format(CLASSES[index - 1], confidence * 100)
            cv2.rectangle(frame, coord, color=(0, 255, 0))
            cv2.putText(frame, label, (coord[0] + coord[3], coord[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Frozen model detection", frame)
    return frame
