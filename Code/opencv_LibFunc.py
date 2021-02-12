import numpy as np
import cv2
import matplotlib.pyplot as plt
from UndistortImage import UndistortImage
from ReadCameraModel import ReadCameraModel
import glob
x_plot = []
z_plot = []
x = 0
N = np.zeros((4,4))
h = np.eye(4)
c_position = np.zeros((3, 1))
c_rotation = np.eye(3)
data = glob.glob("Oxford_dataset/stereo/centre/*.png")
data.sort()
data_img = [cv2.imread(img, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) for img in data]

def data_prep(data_img):
    img_1 = cv2.cvtColor(data_img[x], cv2.COLOR_BAYER_GR2BGR)
    img_2 = cv2.cvtColor(data_img[x + 1], cv2.COLOR_BAYER_GR2BGR)
    fx, fy, cx, cy, G_C_1, LUT1 = ReadCameraModel('Oxford_dataset/model')
    undimg_1 = UndistortImage(img_1, LUT1)
    undimg_1 = cv2.cvtColor(undimg_1, cv2.COLOR_BGR2GRAY)
    imgeqi_1 = cv2.equalizeHist(undimg_1)
    fx, fy, cx, cy, G_C_2, LUT2 = ReadCameraModel('Oxford_dataset/model')
    undimg_2 = UndistortImage(img_2, LUT2)
    undimg_2 = cv2.cvtColor(undimg_2, cv2.COLOR_BGR2GRAY)
    imgeqi_2 = cv2.equalizeHist(undimg_2)

    return fx, fy, cx, cy, imgeqi_1, imgeqi_2


while x < (len(data_img) - 1):
    fx, fy, cx, cy, imgeqi_1, imgeqi_2 = data_prep(data_img)
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]) 
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgeqi_1,None)
    kp2, des2 = orb.detectAndCompute(imgeqi_2,None)
    BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf_match = BF.match(des1,des2)
    bf_match = sorted(bf_match, key = lambda x:x.distance)

    g_pts = []
    for match in bf_match:
        g_pts.append(match)
        
    pts1 = np.float32([kp1[match.queryIdx].pt for match in g_pts])
    pts2 = np.float32([kp2[match.trainIdx].pt for match in g_pts])

    E,mask = cv2.findEssentialMat(pts2, pts1, K, cv2.FM_RANSAC, 0.999, 1.0, None)
    pts, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)
    
    c_position += c_rotation.dot(t) 
    c_rotation = R.dot(c_rotation)
    x_plot.append(c_position[0,0])
    z_plot.append(c_position[2,0])
    print(x)
    x+= 1

final = []
for i in range(len(x_plot)):
    final.append((x_plot[i],z_plot[i]))
plt.scatter(x_plot,z_plot)
plt.show()

