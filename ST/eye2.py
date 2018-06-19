
import cv2


img = cv2.imread('C:/Users/bharat97/Desktop/SUMMER_2k18/01_test.tif')
#img = cv2.GaussianBlur(image1,(3,3),0.3,0.3)
#img = cv2.bilateralFilter(img,0,0.5,0.5)	
#img = cv2.medianBlur(img,3)


img[:,:,0] = 0
img[:,:,2] = 0
cv2.imshow('original',img)




edges = cv2.Canny(img,25,48)

sobelx = cv2.Sobel(edges,cv2.CV_32F,1,0,ksize=3)
sobely = cv2.Sobel(edges,cv2.CV_32F,0,1,ksize=3)
cv2.imshow('sobelx',sobelx)
cv2.imshow('sobely',sobely)

cv2.imshow('edges',edges)

cv2.waitKey(0)
cv2.destroyAllWindows()