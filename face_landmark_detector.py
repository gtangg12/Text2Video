import numpy as np
import cv2 as cv
import dlib

data_dir = './face_landmark_data/'

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(f'{data_dir}/shape_predictor_68_face_landmarks.dat')

# make 5 landmarks from 68 landmarks
# left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner
FIVE_LANDMARKS = [range(36, 42), range(42, 48), [30, 33], [48, 60], [54, 64]]
LIP_LANDMARKS = range(49, 68)

def average_landmarks(landmarks, iter):
    xa, ya = 0, 0
    for n in iter:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        xa += x
        ya += y
    xa /= len(iter)
    ya /= len(iter)
    return int(xa), int(ya)

def landmarks_from_image(img, landmark_grps):
    face = face_detector(img, 1)[0] #get first face using [0]

    landmarks = []
    landmarks = landmark_detector(img, face)
    for iter in landmark_grps:
        x, y = average_landmarks(landmarks, iter)
        # draw circle on image at landmark
        cv.circle(img, (x, y), 3, (255, 255, 0), -1)
        landmarks.append((x, y))
    return landmarks, img

