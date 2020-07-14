import numpy as np
import cv2
import extract_E_matrix as rep
from decimal import Decimal


with open("ground-truth.txt",'r') as f:
    y = f.readlines()

data = []
for line in y:
    x = line.split(' ')
    for i in x:
        data.append(float(i))
data = np.array(data)
data = data.reshape(801,12)
file = open("your_result_curr.txt","w+")
original_translation = np.zeros([801,3],dtype = float)
original_translation[:,0] = data[:,3]
original_translation[:,1] = data[:,7]
original_translation[:,2] = data[:,11]
a = np.array([[1.000000e+00, -1.822835e-10, 5.241111e-10, -5.551115e-17],
             [-1.822835e-10 ,9.999999e-01, -5.072855e-10, -3.330669e-16],
             [5.241111e-10, -5.072855e-10 ,9.999999e-01 ,2.220446e-16]])
t = a[:,3]
a = a.reshape(1,12)
translation_final = np.zeros((3, 1))
Rotation_final = np.eye(3)
scale =1.0
Rotation = np.zeros((3,3))
predicted_translation = np.zeros((3,1))
for i in range(800):
    Rotation,predicted_translation = rep.Find_R_and_t(i,i+1)
    scale = np.sqrt((original_translation[i+1][0] - original_translation[i][0])**2 + (original_translation[i+1][2] - original_translation[i][2])**2)
    predicted_translation = predicted_translation.reshape(3,1)
    translation_final  += (Rotation_final.dot(predicted_translation)) *scale
    Rotation_final = Rotation.dot(Rotation_final)
    a = np.zeros((3,4))
    a[:,0] = Rotation[:,0]
    a[:,1] = Rotation[:,1]
    a[:,3] = Rotation[:,2]
    a[:,3] = translation_final[:,0]
    a = a.reshape((1,12))
    for j in range(12): 
        file.write(str('%.6e'%Decimal(a[0][j])))
        if (j<11):
            new=" "
            file.write(new)
    file.write("\n")
    print("Current image processing :",i+1)

    
for j in range(12): 
        file.write(str('%.6e'%Decimal(a[0][j])))
        if (j<11):
            new=" "
            file.write(new)
file.write("\n")
print("Current image processing :",801)
