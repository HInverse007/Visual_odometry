import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from numpy import linalg as LA

# evo_traj kitti your_result.txt --ref ground-truth.txt -va --plot --plot_mode xz

path="./mr19-assignment2-data/images/000"
extension=".png"

K=np.array([[7.215377000000e+02,0.000000000000e+00,6.095593000000e+02],
            [0.000000000000e+00,7.215377000000e+02,1.728540000000e+02],
            [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])


def ransacc(img1,img2):
    sift=cv.xfeatures2d.SIFT_create()
    key_points=sift.detect(img1)
    pts1=np.array([x.pt for x in key_points],dtype=np.float32)
    #print(pts1)
    pts2,status,_=cv.calcOpticalFlowPyrLK(img1,img2,pts1,None)
    status = status.reshape(status.shape[0])
    pts1 = pts1[status == 1]
    pts2 = pts2[status == 1]
    [x,y]=pts1.shape
    tmp=np.hstack((pts1,np.ones((x,1))));
    Fm=np.zeros((3,3))
    ma=0
    for ii in range(0,151):
        idx=np.random.choice(x,8)
        p1=pts1[idx]
        p2=pts2[idx]
        svdMat=np.zeros((8,9))
        for i in range(0,8):
            svdMat[i][0]=p1[i][0]*p2[i][0]
            svdMat[i][1]=p1[i][1]*p2[i][0]
            svdMat[i][2]=p2[i][0]
            svdMat[i][3]=p1[i][0]*p2[i][1]
            svdMat[i][4]=p1[i][1]*p2[i][1]
            svdMat[i][5]=p2[i][1]
            svdMat[i][6]=p1[i][0]
            svdMat[i][7]=p1[i][1]
            svdMat[i][8]=1

        u,s,vh =LA.svd(svdMat)
        FMat=vh[8,:]
        FMat=FMat.reshape(3,3)
        u,s,vh =LA.svd(FMat)
        s[2]=0
        s=np.diag(s)
        ak=np.matmul(u,s)
        FMat1=np.matmul(ak,vh)
        count=0;
        #return ans_ind
        h=tmp.T
        outlierCheck=np.matmul(np.matmul(tmp,FMat1),h)
        #print(outlierCheck.shape)
        diag=np.diagonal(outlierCheck)
        #print(diag.shape)
        x1=diag.shape
     
        for i in range(0,x1[0]):
            if(np.absolute(diag[i])<1):
                count=count+1
        
        if count>ma:
            ma=count
            ans_ind=idx
            Fm=FMat1

    FMat1=Fm
    E=K.T.dot(FMat1.dot(K))
    R=np.zeros((3,3))
    t=np.zeros(3)
    cv.recoverPose(E,p2,p1,K,R,t) 
    return R,t


def Find_R_and_t(i1,i2):
    if(i1<10):
        f_path1=(path+"00"+str(i1)+extension)
    elif(i1<100):
        f_path1=(path+"0"+str(i1)+extension)
    else:
        f_path1=(path+str(i1)+extension)
    if(i2<10):
        f_path2=(path+"00"+str(i2)+extension)
    elif(i2<100):
        f_path2=(path+"0"+str(i2)+extension)
    else:
        f_path2=(path+str(i2)+extension)

    img1=Image.open(f_path1)
    img2=Image.open(f_path2)
    img1=np.array(img1)
    img2=np.array(img2)
    R,t=ransacc(img1,img2)
    return R,t
