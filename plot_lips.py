import numpy as np
import cv2 
from tqdm import tqdm
sample_lips = np.load('downsample_preds.npy')
sample_lips = sample_lips.reshape(-1, 20, 2)
frames = range(436, 858)
for f in tqdm(frames):
    lip_points = sample_lips[f-436]
    im = cv2.imread(f'/media/william/DATA/6869/frontalized/output_frame_{f}.jpg')
    #print(lip_points)
    #cv2.rectangle(im, (80, 130), (144, 180), (0,0,0), cv2.FILLED)
    for point in lip_points:
        cv2.circle(im, (round(point[0]), round(point[1])), 2, (255, 0, 0), cv2.FILLED)
    cv2.imwrite(f'/media/william/DATA/6869/output_rnn/frame_{f}.jpg', im)
