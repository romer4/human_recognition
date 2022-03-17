import cv2
import imutils
import numpy as np
import os.path
# HOGCV = cv2.HOGDescriptor()
# HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

body_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_fullbody.xml")
# body_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_lowerbody.xml")

def detect(frame):# era para detectar pessoas, porém não funciona tão bem assim
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(grayFrame, 1.1, 3)
    person = 1
    for x,y,w,h in bodies:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('Bodies', frame)
    return frame

# def NMS(boxes):
#     if len(boxes) == 0: return []

#     if boxes.dtype.kind == "i": boxes = boxes.astype("float")

def ObjectDetectionCaffeByPath(path, minConfidence=0.4):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    print("loading model...")
    net = cv2.dnn.readNetFromCaffe(r"./data/MobileNetSSD_deploy.prototxt.txt", r"./data/MobileNetSSD_deploy.caffemodel")

    frame = cv2.imread(path)
    if not os.path.isfile(path):
        print("Imagem não achada")
        return
    frame = imutils.resize(frame, width=min(500, frame.shape[1]))
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    # print(np.arange(0, detections.shape[2]))
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > minConfidence:
            index = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (x1, y1, x2, y2) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[index], confidence * 100)

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[index], 1)

            text_y = 0
            if y1 - 15 > 15: text_y = y1 - 15
            else: text_y = y1 + 15

            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[index], 1)
    
    cv2.imshow("Objects?", frame)
    key = cv2.waitKey(0) & 0xff
    cv2.destroyAllWindows()


def ObjectDetectionFrozenInferenceByPath(path):#detecta vários objetos e pessoas muito bem
    with open("./datasets/coco.names", "rt") as file:
        CLASSES = file.read().rstrip("\n").split("\n")
    print(CLASSES)

    if not os.path.isfile(path):
        print("Imagem não achada")
        return

    frame = cv2.imread(path)
    frame = imutils.resize(frame, width=min(500, frame.shape[1]))

    net = cv2.dnn_DetectionModel(os.path.abspath("./datasets/frozen_inference_graph.pb"), os.path.abspath("./datasets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"))
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    indexs, confidences, coords = net.detect(frame, confThreshold=0.5)
    print(len(CLASSES))
    for index, confidence, coord in zip(indexs.flatten(), confidences.flatten(), coords):
        if CLASSES[index - 1] == "person":
            cv2.rectangle(frame, coord, color=(0, 255, 0))
            label = "{}: {:.2f}%".format(CLASSES[index - 1], confidence * 100)
            cv2.putText(frame, label, (coord[0] + 10, coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
    
    cv2.imshow("Frozen model detection", frame)
    cv2.waitKey(0)