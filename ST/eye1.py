import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from skimage import morphology

# 500 x 250
img = cv2.imread('C:/Users/bharat97/Desktop/od/drishtiGS_010_1_1.png')
#img = cv2.GaussianBlur(image1,(3,3),0.3,0.3)
#img = cv2.bilateralFilter(img,0,0.5,0.5)	
#img = cv2.medianBlur(img,3)


img[:,:,0] = 0
img[:,:,2] = 0
cv2.imshow('original',img)


#edges = cv2.Canny(img,25,48)

sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)

#cv2.imshow('edges',edges)

#sobelx = cv2.Sobel(edges,cv2.CV_32F,1,0,ksize=3)
#sobely = cv2.Sobel(edges,cv2.CV_32F,0,1,ksize=3)




sx = np.multiply(sobelx[:,:,1], sobelx[:,:,1])
sy = np.multiply(sobely[:,:,1], sobely[:,:,1])
sxy = np.multiply(sobelx[:,:,1], sobely[:,:,1])

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

cohbool = np.ones((w, h), dtype=bool)

for i in range(w):
    for j in range(h):
        if erosion[i][j] == 255 and coherence[i][j] > 0.5 and coherence[i][j] < 0.9:
            cohbool[i][j] = True
        else:
            cohbool[i][j] = False
            
coherence = coherence
coh = coherence.astype(np.uint8)
labels = morphology.label(cohbool)

final = morphology.remove_small_objects(labels, min_size=30, connectivity=1)
img3 = np.zeros((final.shape)) # create array of size cleaned
img3[final > 0] = 255 
img3= np.uint8(img3)

cv2.imshow('img3',img3)
cv2.imshow('coherence',coherence)




#
#
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)
cv2.imshow('phase',phase)



cv2.waitKey(0)
cv2.destroyAllWindows()
#
#maxi = mag
#
#for i in range(0,w):
#    for j in range(0,h):
#        for  p in range(0,3):
#            for q in range(0,3):
#                if i-1+p >= 0 and j-1+q >= 0 and i-1+p < w and j-1+q < h:
#                    if maxi[i][j] < mag[i-1+p][j-1+q]:
#                        maxi[i][j] = 0
#mag = maxi

image = plt.imread("C:/Users/bharat97/Desktop/od/drishtiGS_010_1_1.png")


#
#X = range(500)
#ax.imshow(image, extent=[0, w*3, 0, h*3])
#ax.plot(X, X, '--', linewidth=1, color='white')

im = plt.imread("C:/Users/bharat97/Desktop/od/drishtiGS_010_1_1.png")

lines = []

for i in range(0,w,3):
    for j in range(0,h,3):
        if (mag[i][j] > 0.5 or mag[i][j] < -0.5) and mag[i][j] < 3 and erosion[i][j] == 255 and img3[i][j] >1:
            a1 = i/w + (0.000*eigvalues[i][j][1]*eigvectors[i][j][0][0])/w
            b1 = j/w - (0.000*eigvalues[i][j][1]*eigvectors[i][j][0][1])/w
            a2 = i/h - (0.000*eigvalues[i][j][1]*eigvectors[i][j][0][0])/h
            b2 = j/h + (0.000*eigvalues[i][j][1]*eigvectors[i][j][0][1])/h
            lines.append([(b1,1-a1),(b2,1-a2)]) 

lc = mc.LineCollection(lines, linewidths=1)
fig, ax = plt.subplots()
ax.imshow(im,extent=[0,w/h,0,1.0])
ax.add_collection(lc)

#a1 = (i-eigvalues[i,j,1]*eigvectors[i,j,0,0])/w
#                    b1 = (j-eigvalues[i,j,1]*eigvectors[i,j,0,1])/w
#                    a2 = (i+eigvalues[i,j,1]*eigvectors[i,j,0,0])/h
#                    b2 = (j+eigvalues[i,j,1]*eigvectors[i,j,0,1])/h
