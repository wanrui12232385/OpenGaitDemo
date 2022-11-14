import cv2
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def convertPkl2Webp(pklfilepath):
    count = 0
    with open(pklfilepath, "rb") as pickle_file:
        for imgpk in pickle.load(pickle_file):
            img = np.array(imgpk)
            cv2.imwrite("/home/jdy/Gaitdateset/Image/tmp/pklImg-{}.png".format(count), img)
            count+=1
    # im  = Image.fromarray(np.around(img*255).astype(np.int16))
    # im.save("test.webp", quality=100)

def run():
    convertPkl2Webp("/home/jdy/Gaitdateset/Image/tmp-pkl/Gaitdateset/Image/out_afterPaddleSeg/out_afterPaddleSeg.pkl")
    # #读入lena图形
    # imgname = 'test.webp'
    # img = cv2.imread(imgname)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度处理图像

    # #使用opencv中封装的sift方法
    # sift = cv2.SIFT_create()

    # keyPoints, des = sift.detectAndCompute(img, None)  #keyPoint是关键点的信息，des是关键点对应的描述信息

    # #把提取到的关键点，在原图中标出
    # show_img = cv2.drawKeypoints(img, keyPoints, img, color=(255,0,255)) #画出特征点，并显示为红色圆圈
    # point_set = set()
    # for key_point in keyPoints:
    #     x = int(round(key_point.pt[0]))
    #     y = int(round(key_point.pt[1]))
    #     point_set.add((x, y))
    # #显示
    # cv2.imshow("SIFT", show_img)
    # cv2.waitKey(0)



if __name__ == '__main__':
    run()
