import cv2
import imutils
from detectBody import detect
from detectFace import detectFace

def detectByPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return
    print('Detecting people...')
    while video.isOpened():
        #check is True if reading was successful 
        check, frame =  video.read()
        if check:
            frame = imutils.resize(frame , width=min(800, frame.shape[1]))
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            # key = cv2.waitKey(0) & 0xff
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

def DetectFacesInVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False: 
        print("Vídeo não foi achado")
        return

    while video.isOpened():
        check, frame = video.read()
        if check == True:
            frame = imutils.resize(frame , width=min(320, frame.shape[1]))
            frame = detectFace(frame)
            
            if writer is not None:
                writer.write(frame)
            
            # key = cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: break
    
    video.release()
    cv2.destroyAllWindows()