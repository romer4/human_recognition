from os import path
import cv2
# from imageai.Detection import ObjectDetection
import imutils
import numpy as np
import argparse
import glob
# from detectCamera import *
# from detectVideo import *
# from detectImage import *
# from detectFace import *
from detectBody import *

from fileFormats import *
from detectionMethods import *
from filters import *

from log import *

# def humanDetector(args):
#     image_path = args["image"]
#     video_path = args['video']
#     if str(args["camera"]) == 'true' : camera = True 
#     else : camera = False
#     writer = None
#     if args['output'] is not None and image_path is None:
#         writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
#     if camera:
#         print('[INFO] Opening Web Cam.')
#         # detectByCamera(writer)
#     elif video_path is not None:
#         print('[INFO] Opening Video from path.')
#         detectByPathVideo(video_path, writer)
#     elif image_path is not None:
#         print('[INFO] Opening Image from path.')
#         detectByPathImage(image_path, args['output'])

# detectByPathVideo(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data\video1.mp4", writer)
# detectByCamera(writer)

# detectFaceByPath(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data\yt.jpg")
# detectFaceByPath(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data\facescover.jpg")
# detectFaceByPath(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data\group2.jpg")
# detectFaceByPath(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data\selfie_group.jpg")

# detectByPathImage(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data\group.jpg", r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data_output\output.jpg")
# DetectFacesInVideo(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data\facevideo.mp4", writer)

# paths = glob.glob(r"data\videos\*")

# for path in paths:
#     video(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\{}".format(path), writer, detectObjects)

# ObjectDetectionByPath(r"C:\Users\Romera\OneDrive\Projetos\Python\human_detection\data\tv.jpg")
# ObjectDetectionFrozenInferenceByPath(os.path.abspath(r"./data/object_detection.jpeg"))

# image(r"./data/edu.png", detectFace, r"./data/obama2.jpg")
# image(r"./data/obama1.jpeg", detectFace, r"./data/obama2.jpg")
# image(r"./data/selfie_group.jpg", detectFace, r"./data/obama2.jpg")

writer = cv2.VideoWriter(r"./data_output/output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 1, (600,600))
# video2Image(r"./data/facevideo.mp4", writer, detectFace, catch_frame=1)

video(r"./data/video1.mp4", writer, detectObjects)
# camera(writer)

# print(compareFaces(r"./data/obama1.jpeg", r"./data/obama2.jpg"))
