import cv2
import numpy as np
import matplotlib.pyplot as plt

# 500 x 250
img = cv2.imread('C:/Users/bharat97/Desktop/SUMMER_2k18/03_test.tif')

img[:,:,0] = 0
img[:,:,2] = 0
#cv2.imshow('original',img)

sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
sx = np.multiply(sobelx[:,:,1], sobelx[:,:,1])
sy = np.multiply(sobely[:,:,1], sobely[:,:,1])
sxy = np.multiply(sobelx[:,:,1], sobely[:,:,1])

phase=cv2.phase(sobelx,sobely,angleInDegrees=True)
mag = (sx + sy)

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

st = np.zeros((w,h,2,2))
eigvalues=np.zeros((w,h,2))
eigvectors =np.zeros((w,h,2,2))
for i in range(w):
    for j in range(h):
        st[i,j,0,0] = sx[i,j]
        st[i,j,0,1] = sxy[i,j]
        st[i,j,1,0] = sxy[i,j]
        st[i,j,1,1] = sy[i,j]
        eigvalues[i,j],eigvectors[i,j] = np.linalg.eigh(st[i,j])
#        
#extra = np.zeros((w,h))
#        
#for i in range(w):
#    for j in range(h):
#        
#        extra[i,j] = eigvalues[i,j,1]
#              
#cv2.imshow('extra',extra)
#p = np.asarray(mag).astype('int8')
#w,h = mag.shape
#y, x = np.mgrid[0:h:500j, 0:w:500j]
#[dy, dx] = np.gradient(p)
#q = np.multiply(dx, dx) + np.multiply(dy, dy)
#cv2.imshow('q',q)



#plt.figure()
#plt.title('Arrows scale with plot width, not view')
#Q = plt.quiver(x[::3], y[::3], sx[::3], sy[::3], units='width')
#qk = plt.quiverkey(Q, 0.1, 0.1, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                   coordinates='figure')
#[dy, dx] = np.gradient(p)
#skip = (slice(None, None, 3), slice(None, None, 3))
#
#fig, ax = plt.subplots()
#im = ax.imshow(dx, extent=[x.min(), x.max(), y.min(), y.max()])
#ax.quiver(x[skip], y[skip], dx[skip], dy[skip])
#
#ax.set(aspect=1, title='Quiver Plot')
#plt.show()




#cv2.imshow('sobelx',sobelx)
#cv2.imshow('sobely',sobely)
#cv2.imshow('phase',phase)
#cv2.imshow('mag',mag)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()