import numpy as np
import cv2 as cv
import dlib

data_dir = 'face_landmark_data/'

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(f'{data_dir}/shape_predictor_68_face_landmarks.dat')

img = cv.imread(f'{data_dir}/down.png')
print(img.shape)
img = cv.resize(img, (500, 750))

faces = face_detector(img, 1)

landmark_coords = []

for k, d in enumerate(faces):
    landmarks = landmark_detector(img, d)
    for n in range(48, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_coords.append((x, y))
        # draw circle on image at landmark
        cv.circle(img, (x, y), 2, (255, 255, 0), -1)

cv.imwrite('down_out.png', img)
