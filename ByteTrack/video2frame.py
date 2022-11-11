from turtle import width
from types import CoroutineType
import cv2
import sys
import pandas as pd
import PIL
import numpy as np

def video2frame(video_path, frame_save_path, label_path):
    videocap = cv2.VideoCapture(video_path)
    df = pd.read_csv(label_path, sep=',', header=None)
    count = 0
    for i in range(0,df.__len__()):
        if df[6][i]<0.5 :
            continue
        frame_id = df[0][i]
        people_id = df[1][i]
        x = df[2][i]
        y = df[3][i]
        width = df[4][i]
        height = df[5][i]
        videocap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        flag, frame = videocap.read()
        if flag == False:
            break
            
        if x < 0:
            x = -x
        x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
        w, h = x2 - x1, y2 - y1
        new_w = 512 * w // h
        tmp = frame[y1: y2, x1: x2, :]
        print(count, frame_id, people_id, x, y, width, height)

        tmp = cv2.resize(tmp, (new_w, 512))
        img = np.zeros((512, 512, 3))
        img[0:512, 0:new_w, :] = tmp[:, :, :]
        cv2.imwrite("{}/outTmp-{}-{}-{}.jpg".format(frame_save_path, count, people_id, frame_id), tmp)
        cv2.imwrite("{}/{}-outImg-{}-{}.jpg".format(frame_save_path, count, people_id, frame_id), img)
        count+=1
        # cv2.imwrite("./test.jpg", img)
        #break
        #cv2.imencode('.jpg',frame)[1].tofile(frame_save_path+'/'+str(frame_id)+'.jpg')
    #分割
        
  
v = '/home/jdy/ByteTrack/videos/palace.mp4'
f = '/home/jdy/ByteTrack/cutvideo2frame'
l = '/home/jdy/ByteTrack/YOLOX_outputs/yolox_x_mix_det/track_vis/2022_11_08_19_18_13.txt'

video2frame(v, f ,l)