import cv2
import numpy as np

# 500 x 250
img = cv2.imread('C:/Users/bharat97/Desktop/SUMMER_2k18/03_test.tif')
print(img[:,:,1])
img[:,:,0] = 0
img[:,:,2] = 0
cv2.imshow('original',img)
#
#sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
#sobely= cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
#phase=cv2.phase(sobelx,sobely,angleInDegrees=True)
#
#cv2.imshow('sobelx',sobelx)
#cv2.imshow('sobely',sobely)
#cv2.imshow('phase',phase)

cv2.waitKey(0)
cv2.destroyAllWindows()