# import os.path
# import cv2


# # ---------------------------

# def from_video_get_img(img_path):
#     # 创建存放结果图片的文件
#     # save_file_name = video_path.split('/')[-1].split('.')[0]
#     save_fgmask_path = '/home/jdy/Gaitdateset/Image/final_Img'
#     if not os.path.exists(save_fgmask_path):
#         os.makedirs(save_fgmask_path)


#     fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#     fgmask_img_list = []

#     # ret, frame = cap.read()
#     frame = cv2.imread(img_path) 
#     fgmask = fgbg.apply(frame)

#     fgmask_img_list.append(fgmask)
#     print(len(fgmask_img_list))

#     i=0
#     # cv2.imwrite(save_fgmask_path + '1.jpg', fgmask_img_list[i])
#     # cv2.imwrite("{}/{}-outImg-{}-{}.jpg".format(frame_save_path, count, people_id, frame_id), img)
#     cv2.imwrite("{}/{}-outImg.jpg".format(save_fgmask_path, i), frame)

# if __name__ == "__main__":
#     from_video_get_img('Gaitdateset/Image/out_afterPaddleSeg/portrait_shu_v2.jpg')

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/jdy/Gaitdateset/Image/out_afterPaddleSeg/portrait_shu_v2.jpg')
        #   /home/jdy/Gaitdateset/Image/out_afterPaddleSeg/portrait_shu_v2.jpg
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
grayImage = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(grayImage,120,255,cv2.THRESH_BINARY_INV)

plt.imshow(thresh, cmap="gray", vmin=0, vmax=255),plt.show()

"""

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(

thresh,
None,
None,
None,
8,
cv2.CV_32S)
sizes = stats[1:, -1]
img2 = np.zeros((labels.shape), np.uint8)
for i in range(0, nlabels - 1):
if sizes[i] >= 25: #filter small dotted regions
img2[labels == i + 1] = 255
thresh = cv2.bitwise_not(img2)
plt.imshow(thresh, cmap="gray", vmin=0, vmax=255),plt.show()

"""

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv2.dilate(opening,kernel,iterations=20)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(image,markers)
image[markers == -1] = [255,0,0]
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (10,10,665,344)
cv2.grabCut(image,markers,rect,bgdModel,fgdModel,50,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((markers==2)|(markers==0),0,1).astype('uint8')
image = image*mask2[:,:,np.newaxis]
save_fgmask_path = "/home/jdy/Gaitdateset/Image/final_Img"
i=0
plt.imshow(image, cmap="gray", vmin=0, vmax=255),plt.show()
plt.savefig("{}/{}-outImgplt.jpg".format(save_fgmask_path, i))


# save_fgmask_path = "/home/jdy/Gaitdateset/Image/final_Img"
# i = 0
# plt.savefig("{}/{}-outImgplt.jpg".format(save_fgmask_path, i), image)
# cv2.imwrite("{}/{}-outImg.jpg".format(save_fgmask_path, i), image)