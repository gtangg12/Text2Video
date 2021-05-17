import numpy as np
import cv2 as cv
import dlib

data_dir = 'face_landmark_data/'

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(f'{data_dir}/shape_predictor_68_face_landmarks.dat')

img = cv.imread(f'{data_dir}/image_data/me2.jpg')
print(img.shape)
#img = cv.resize(img, (500, 750))

faces = face_detector(img, 1)

landmark_coords = []

# make 5 landmarks from 68 landmarks
# left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner
five_landmarks = [range(36, 42), range(42, 48), [30, 33], [48, 60], [54, 64]]

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


for k, d in enumerate(faces):
    landmarks = landmark_detector(img, d)
    '''
    for n in range(68): #[36, 45, 30, 57], range(18, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_coords.append((x, y))
        # draw circle on image at landmark
        cv.circle(img, (x, y), 3, (255, 255, 0), -1)
    print(landmark_coords)
    '''
    for iter in five_landmarks:
        x, y = average_landmarks(landmarks, iter)
        cv.circle(img, (x, y), 3, (255, 255, 0), -1)
        print(x, y)

cv.imwrite('me2.png', img)
