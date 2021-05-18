import numpy as np
import cv2 as cv
import dlib
import os
from multiprocessing import Pool
import gc
from tqdm import tqdm
import pandas as pd

data_dir = 'face_landmark_data'

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(os.path.join( \
    os.path.dirname(__file__), data_dir, 'shape_predictor_68_face_landmarks.dat'))
print(os.path.join( \
    os.path.dirname(__file__), data_dir, 'shape_predictor_68_face_landmarks.dat'))
# make 5 landmarks from 68 landmarks
# left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner
FIVE_LANDMARKS = [range(36, 42), range(42, 48), [30, 33], [48, 60], [54, 64]]
LIP_LANDMARKS = map(lambda x: [x], range(49, 68))

def average_landmarks(landmarks, iters):
    xa, ya = 0, 0
    for n in iters:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        xa += x
        ya += y
    xa /= len(iters)
    ya /= len(iters)
    return int(xa), int(ya)

def expand_rect(rect):
    #expands face detector results to fit face
    return dlib.rectangle(
        rect.left() - rect.width() // 3,
        rect.top() - rect.height() * 2 // 3,
        rect.right() + rect.width() // 3,
        rect.bottom() + rect.height() // 3,
    )

def landmarks_from_image(img, landmark_grps):
    faces = face_detector(img, 1)
    if not faces:
        return np.zeros((5,2))  #need to replace hardcoded value
    face = max(faces, key=lambda f: f.area())
    #face = dlib.rectangle(480, 0, 800, 360)

    all_landmarks = []
    landmarks = landmark_detector(img, face)

    for iters in landmark_grps:
        x, y = average_landmarks(landmarks, iters)
        # draw circle on image at landmark
        #cv.circle(img, (x, y), 3, (255, 255, 0), -1)
        all_landmarks.append((x, y))
    return all_landmarks

def process_vid_part(process_id):
    cap = cv.VideoCapture(vid_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_jumps * process_id)

    all_landmarks = []
    for _ in tqdm(range(frame_jumps), position=process_id):
        frame = cap.read()[1]
        all_landmarks.append(landmarks_from_image(frame[:,:,::-1], landmark_grps))
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

    np.save('xAAmF3H0-ek_landmarks_pooled', np.array(landmarks_from_video()))
    """
    for i in range(3):
        img = cv.imread(f'./face_landmark_data/image_data/test{i+1}.jpg')
        lm = landmarks_from_image(img, FIVE_LANDMARKS)
        pd.DataFrame(lm).to_csv(f'test{i+1}.txt', sep=' ', index=False, header=False)
    """