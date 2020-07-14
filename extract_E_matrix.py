import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

path = "./mr19-assignment2-data/images/000"
extension = ".png"

K = np.array([[7.215377000000e+02,0.000000000000e+00,6.095593000000e+02],
    [0.000000000000e+00,7.215377000000e+02,1.728540000000e+02],
    [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])

def Find_R_and_t(i1,i2):
    if(i1<10):
        f_path1 = (path+"00"+str(i1)+extension)
    elif(i1<100):
        f_path1 = (path+"0"+str(i1)+extension)
    else:
        f_path1 = (path+str(i1)+extension)

    if(i2<10):
        f_path2 = (path+"00"+str(i2)+extension)
    elif(i2<100):
        f_path2 = (path+"0"+str(i2)+extension)
    else:
        f_path2 = (path+str(i2)+extension)

    img1=Image.open(f_path1)
    img2=Image.open(f_path2)
    img1=np.array(img1)
    img2=np.array(img2)

    sift=cv.xfeatures2d.SIFT_create()
    key_points=sift.detect(img1)
    pts1 = np.array([x.pt for x in key_points],dtype=np.float32)
    pts2,status,_=cv.calcOpticalFlowPyrLK(img1,img2,pts1,None)
    status = status.reshape(status.shape[0])
    pts1 = pts1[status == 1]
    pts2 = pts2[status == 1]
    R = np.zeros((3,3))
    t = np.zeros(3)
    E, mask = cv.findEssentialMat(pts2, pts1, K, cv.RANSAC, 0.999, 1.0, None)
    cv.recoverPose(E, pts2, pts1, K, R,t)
    return R,t


