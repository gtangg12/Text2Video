import cv2 as cv
import os
from tqdm import tqdm

vid_path = './obama_addresses/xAAmF3H0-ek_video.mp4'
os.makedirs('/media/william/DATA/6869/xAAmF3H0-ek/', exist_ok=True)

cap = cv.VideoCapture(vid_path)
tot = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
for i in tqdm(range(tot)):
    frame = cap.read()[1]
    frame = frame[2:340, 480:800] 
    cv.imwrite(f'/media/william/DATA/6869/xAAmF3H0-ek/frame_{i}.jpg', cv.resize(frame, (224, 224)))
