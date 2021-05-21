import numpy as np
import cv2 
from tqdm import tqdm
sample_lips = np.load('downsample_preds_rts.npy')
sample_times = np.load('downsample_times_rts.npy')
sample_lips = sample_lips.reshape(-1, 18, 2)
#frames = range(13568, 15261)
for t, lip in tqdm(zip(sample_times, sample_lips)):
    lip_points = lip
    f = round(30*t)
    im = cv2.imread(f'/media/william/DATA/6869/frontalized/output_frame_{f}.jpg')
    #print(lip_points)
    #cv2.rectangle(im, (80, 130), (144, 180), (0,0,0), cv2.FILLED)
    for point in lip_points:
        cv2.circle(im, (round(point[0]), round(point[1])), 2, (255, 0, 0), cv2.FILLED)
    cv2.imwrite(f'/media/william/DATA/6869/output_rnn/rts2/frame_{f}.jpg', im)
