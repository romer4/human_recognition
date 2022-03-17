import cv2
import imutils
import os.path
from detectionMethods import *
from log import *

#se o param known_image_path for passado, o filtro de rostos será ativado
def image(path, detector=detectFace, known_image_path=None):
    if not os.path.isfile(path):
        print("Imagem não encontrada! @(")
        return

    frame = cv2.imread(path)
    frame = imutils.resize(frame, width=min(320, frame.shape[1]))

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if known_image_path: 
        known_image_frame = cv2.imread(known_image_path)
        detector(frame, known_image_frame, True)
    else: detector(frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video2Image(path, writer, detector=detectFace, known_image_path=None, catch_frame=0):
    if not os.path.isfile(path):
        print("Vídeo não encontrado! @(")
        return

    video = cv2.VideoCapture(path)
    check, frame = video.read()
    
    if check == False:
        print("O arquivo não é um vídeo")
        return
    
    #loop para pegar o frame pedido
    curr_frame = 0
    while video.isOpened():
        if curr_frame == catch_frame: break
        check, frame = video.read()
        curr_frame += 1
        if writer is not None:
            writer.write(frame)
    ###############################

    log(curr_frame)
    frame = imutils.resize(frame, width=min(320, frame.shape[1]))

    if known_image_path: 
        known_image_frame = cv2.imread(known_image_path)
        detector(frame, known_image_frame, True)
    else: detector(frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def video(path, writer, detector=detectFace):
    if not os.path.isfile(path):
        print("Vídeo não encontrado! @(")
        return

    video = cv2.VideoCapture(path)
    check, frame = video.read()

    if check == False:
        print("O arquivo não é um vídeo")
        return
    
    while video.isOpened():
        check, frame = video.read()
        if check:
            frame = imutils.resize(frame, width=min(320, frame.shape[1]))
            frame = detector(frame)

            if writer is not None:
                writer.write(frame)

            if cv2.waitKey(1) & 0xff == ord("q"):
                break
        else:
            break
    
    video.release()
    cv2.destroyAllWindows()

def camera(writer, detector=detectFace):   
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()
        frame = detector(frame)
        
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
        
    video.release()
    cv2.destroyAllWindows()
