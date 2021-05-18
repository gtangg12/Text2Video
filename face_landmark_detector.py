import numpy as np
import cv2 as cv
import dlib
import os
from multiprocessing import Pool
import gc

data_dir = 'face_landmark_data'

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(os.path.join( \
    os.path.dirname(__file__), data_dir, 'shape_predictor_68_face_landmarks.dat'))

# make 5 landmarks from 68 landmarks
# left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner
FIVE_LANDMARKS = [range(36, 42), range(42, 48), [30, 33], [48, 60], [54, 64]]
LIP_LANDMARKS = map(lambda x: [x], range(49, 68))

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
    face = face_detector(img, 1)
    if not face:
        return np.zeros((5,2))
    face = face[0] #get first face using [0]

    all_landmarks = []
    landmarks = landmark_detector(img, face)

    for iter in landmark_grps:
        x, y = average_landmarks(landmarks, iter)
        # draw circle on image at landmark
        #cv.circle(img, (x, y), 3, (255, 255, 0), -1)
        all_landmarks.append((x, y))
    return all_landmarks

def process_vid_part(process_id):
    cap = cv.VideoCapture(vid_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_jumps * process_id)

    all_landmarks = []
    for _ in range(frame_jumps):
        frame = cap.read()[1]
        landmarks_from_image(frame[:,:,::-1], landmark_grps)
        all_landmarks.append(landmarks_from_image)
    cap.release()

    return all_landmarks

def landmarks_from_video():
    with Pool(num_processes) as p:
        return p.map(process_vid_part, range(num_processes))
    #return list(map(process_vid_part, range(num_processes)))

if __name__ == '__main__':
    vid_path = './obama_addresses/xAAmF3H0-ek_video.mp4'

    cap = cv.VideoCapture(vid_path)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    del cap

    num_processes = 7
    frame_jumps = num_frames // num_processes
    landmark_grps = FIVE_LANDMARKS

    np.save('xAAmF3H0-ek', np.array(landmarks_from_video()))
    

