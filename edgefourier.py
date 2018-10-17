import cv2
import numpy as np

im1 = "C:\\Users\\Ferenc\\Documents\\szakdoga\\it_pyr_test.png"
im = cv2.imread(im1)
imedge = cv2.Canny(im,20,70)
cv2.imshow("edge det",imedge)


a, contours ,hier =  cv2.findContours(imedge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
imgtodraw = np.zeros_like(imedge)

for c in contours:
    if cv2.contourArea(c) > 15:
        cv2.drawContours(imgtodraw,[c],0,(127),2)

cv2.imshow("cont",imgtodraw)

cv2.waitKey(0)