from turtle import width
from types import CoroutineType
import cv2
import sys
import pandas as pd
import PIL
import numpy as np
import os
import argparse

def make_parser():
    parser = argparse.ArgumentParser("Video2frame Demo!")

    # file
    parser.add_argument(
        "-f",
        "--file",
        default='/home/jdy/Gaitdateset/ByteTrack/YOLOX_outputs/yolox_x_mix_det/track_vis/2022_11_08_19_18_13.txt',
        type=str,
        help="pls input your track file",
    )

    # video path
    parser.add_argument(
        "-v", 
        "--video", 
        type=str, 
        default='/home/jdy/Gaitdateset/ByteTrack/videos/palace.mp4', 
        help="pls input your video path")
    return parser


def video2frame(video_path, label_path):
    save_dir_name = video_path.split('/')[-1].split('.')[0]
    frame_save_path = os.getcwd() + "/Image/out_afterByteTrack/" + save_dir_name
    if not os.path.exists(frame_save_path):
        os.makedirs(frame_save_path)
    print(save_dir_name, frame_save_path)
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

        # 159有个x是负数    
        # if x < 0:
        #     x = -x
        x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
        w, h = x2 - x1, y2 - y1
        x1_new = max(0, int(x1 - 0.1 * w))
        x2_new = min(int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(x2 + 0.1 * w))
        y1_new = max(0, int(y1 - 0.1 * h))
        y2_new = min(int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(y2 + 0.1 * h))
        new_w = x2_new - x1_new
        new_h = y2_new - y1_new

        # new_w = 512 * w // h
        tmp = frame[y1_new: y2_new, x1_new: x2_new, :]
        print(count, frame_id, people_id, x, y, width, height, new_w, new_h, int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videocap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        cv2.imwrite("{}/{}-outTmp-{}-{}.jpg".format(frame_save_path, count, people_id, frame_id), tmp)
    
        count+=1
        if count >= 5:
            break
    #分割
        
  
# v = '/home/jdy/Gaitdateset/ByteTrack/videos/palace.mp4'
# l = '/home/jdy/Gaitdateset/ByteTrack/YOLOX_outputs/yolox_x_mix_det/track_vis/2022_11_08_19_18_13.txt'

if __name__ == "__main__":
    args = make_parser().parse_args()
    video2frame(args.video, args.file)











# from turtle import width
# from types import CoroutineType
# import cv2
# import sys
# import pandas as pd
# import PIL
# import numpy as np
# import os

# def video2frame(video_path, frame_save_path, label_path):
#     save_dir_name = video_path.split('/')[-1].split('.')[0]
#     path = os.getcwd() + "/Image/out_afterByteTrack"
#     print(save_dir_name, path)
#     videocap = cv2.VideoCapture(video_path)
#     df = pd.read_csv(label_path, sep=',', header=None)
#     count = 0
#     for i in range(0,df.__len__()):
#         if df[6][i]<0.5 :
#             continue
#         frame_id = df[0][i]
#         people_id = df[1][i]
#         x = df[2][i]
#         y = df[3][i]
#         width = df[4][i]
#         height = df[5][i]
#         videocap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#         flag, frame = videocap.read()
#         if flag == False:
#             break

#         # 159有个x是负数    
#         # if x < 0:
#         #     x = -x
#         x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
#         w, h = x2 - x1, y2 - y1
#         x1_new = max(0, int(x1 - 0.1 * w))
#         x2_new = min(int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(x2 + 0.1 * w))
#         y1_new = max(0, int(y1 - 0.1 * h))
#         y2_new = min(int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(y2 + 0.1 * h))
#         new_w = x2_new - x1_new
#         new_h = y2_new - y1_new

#         # new_w = 512 * w // h
#         tmp = frame[y1_new: y2_new, x1_new: x2_new, :]
#         print(count, frame_id, people_id, x, y, width, height, new_w, new_h, int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videocap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#         cv2.imwrite("{}/{}-outTmp-{}-{}.jpg".format(frame_save_path, count, people_id, frame_id), tmp)
    
#         count+=1
#         if count >= 5:
#             break
#     #分割
        
  
# v = '/home/jdy/Gaitdateset/ByteTrack/videos/palace.mp4'
# f = '/home/jdy/Gaitdateset/Image/out_afterByteTrack'
# l = '/home/jdy/Gaitdateset/ByteTrack/YOLOX_outputs/yolox_x_mix_det/track_vis/2022_11_08_19_18_13.txt'

# video2frame(v, f ,l)