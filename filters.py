import cv2
import face_recognition
import math

# def compareFaces(known_path, unknown_path, threshold=0.6):
#     known_image = face_recognition.load_image_file(known_path)
#     unknown_image = face_recognition.load_image_file(unknown_path)

#     known_encoding = face_recognition.face_encodings(known_image)[0]
#     unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

#     result = face_recognition.compare_faces([known_encoding], unknown_encoding, threshold)

#     percentage = faceDistanceToPercent(face_recognition.face_distance([known_encoding], unknown_encoding)[0])
    
#     return [result[0], percentage]

def compareFaces(known_image, unknown_image, threshold=0.1):
    known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
    unknown_image_rgb = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
    
    known_encoding = face_recognition.face_encodings(known_image_rgb)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image_rgb)
    if len(unknown_encoding) != 0: unknown_encoding = unknown_encoding[0]
    else: return [False, 0]

    result = face_recognition.compare_faces([known_encoding], unknown_encoding, threshold)

    percentage = faceDistanceToPercent(face_recognition.face_distance([known_encoding], unknown_encoding)[0])
    
    return [result[0], percentage]

def faceDistanceToPercent(face_distance, face_threshold=0.6):
    print((1 - face_distance) * 100)

# def faceDistanceToPercent(face_distance, face_match_threshold=0.6):
#     if face_distance > face_match_threshold:
#         range = (1.0 - face_match_threshold)
#         linear_val = (1.0 - face_distance) / (range * 2.0)
#         return linear_val
#     else:
#         range = face_match_threshold
#         linear_val = 1.0 - (face_distance / (range * 2.0))
#         return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))