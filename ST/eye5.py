import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from skimage import morphology

img = cv2.imread('C:/Users/bharat97/Desktop/od/drishtiGS_010_1_1.png',0)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
cl1 = clahe.apply(img)
 
sobelx = cv2.Sobel(cl1,cv2.CV_32F,1,0,ksize=3)
sobely = cv2.Sobel(cl1,cv2.CV_32F,0,1,ksize=3)
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)
cv2.imshow('CLAHE',cl1)
#cv2.imshow('edges',edges)

#sobelx = cv2.Sobel(edges,cv2.CV_32F,1,0,ksize=3)
#sobely = cv2.Sobel(edges,cv2.CV_32F,0,1,ksize=3)




sx = np.multiply(sobelx, sobelx)
sy = np.multiply(sobely, sobely)
sxy = np.multiply(sobelx, sobely)

phase=cv2.phase(sobelx,sobely,angleInDegrees=True)
mag = (sx + sy)

mag = cv2.medianBlur(mag,3)
maxmax = np.max(mag)
sx = np.divide(sx, mag)
sy = np.divide(sy, mag)
sxy = np.divide(sxy, mag)

sx = np.nan_to_num(sx, copy = False)
sy = np.nan_to_num(sy, copy = False)
sxy = np.nan_to_num(sxy, copy = False)

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

w,h = mag.shape

min = np.min(mag)
max = np.max(mag)
mag = mag*255/max

cv2.imshow('mag',mag)

st = np.zeros((w,h,2,2))
eigvalues=np.zeros((w,h,2))
eigvectors =np.zeros((w,h,2,2))
for i in range(w):
    for j in range(h):
        st[i,j,0,0] = sx[i,j]
        st[i,j,0,1] = sxy[i,j]
        st[i,j,1,0] = sxy[i,j]
        st[i,j,1,1] = sy[i,j]
        
st[:,:,0,0] = cv2.GaussianBlur(st[:,:,0,0],(3,3),1,1)
st[:,:,0,1] = cv2.GaussianBlur(st[:,:,0,1],(3,3),1,1)
st[:,:,1,0] = cv2.GaussianBlur(st[:,:,1,0],(3,3),1,1)
st[:,:,1,1] = cv2.GaussianBlur(st[:,:,1,1],(3,3),1,1)

for i in range(w):
    for j in range(h):
        eigvalues[i,j],eigvectors[i,j] = np.linalg.eigh(st[i,j])
        
        
mask = plt.imread("C:/Users/bharat97/Desktop/od/drishtiGS_010_1_1.png")
kernel = np.ones((11,11),np.uint8)
erosion = cv2.erode(mask,kernel,iterations = 1)
        
coherence = np.zeros((w,h))

for i in range(w):
    for j in range(h):
        e1 = eigvalues[i][j][0]
        e2 = eigvalues[i][j][1]
        if e1 == 0 and e2 == 0:
            coherence[i][j] = -1
        else:
            coherence[i][j] = (e1 - e2)**2/(e1 + e2)**2
         
coh_max = np.max(coherence)

            
for i in range(w):
    for j in range(h):
        if coherence[i][j] == -1:
            coherence[i][j]  = coh_max

coh_min = np.min(coherence)
coherence = (coherence-coh_min)/(coh_max - coh_min)
coherence = coherence**0.2

coherence.astype(np.uint8)
coherence = 255*coherence
coh = coherence.astype(np.uint8)

#coherence = cv2.cvtColor(coherence,cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(8,8))
cohclahe = clahe.apply(coh)



cv2.imshow('coherence',coh)
cv2.imshow('cohCLHE',cohclahe)



#
#
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)



cv2.waitKey(0)
cv2.destroyAllWindows()


image = plt.imread("C:/Users/bharat97/Desktop/od/drishtiGS_010_1_1.png")

im = plt.imread("C:/Users/bharat97/Desktop/od/drishtiGS_010_1_1.png")

lines = []

for i in range(0,w,3):
    for j in range(0,h,3):
        if (mag[i][j] > 8 or mag[i][j] < -8) and erosion[i][j] == 255:
            a1 = i/w + (0.000*eigvalues[i][j][1]*eigvectors[i][j][0][0])/w
            b1 = j/w - (0.000*eigvalues[i][j][1]*eigvectors[i][j][0][1])/w
            a2 = i/h - (0.000*eigvalues[i][j][1]*eigvectors[i][j][0][0])/h
            b2 = j/h + (0.000*eigvalues[i][j][1]*eigvectors[i][j][0][1])/h
            lines.append([(b1,1-a1),(b2,1-a2)]) 

lc = mc.LineCollection(lines, linewidths=1)
fig, ax = plt.subplots()
ax.imshow(im,extent=[0,564/585.,0,1.0])
ax.add_collection(lc)

#a1 = (i-eigvalues[i,j,1]*eigvectors[i,j,0,0])/w
#                    b1 = (j-eigvalues[i,j,1]*eigvectors[i,j,0,1])/w
#                    a2 = (i+eigvalues[i,j,1]*eigvectors[i,j,0,0])/h
#                    b2 = (j+eigvalues[i,j,1]*eigvectors[i,j,0,1])/h
